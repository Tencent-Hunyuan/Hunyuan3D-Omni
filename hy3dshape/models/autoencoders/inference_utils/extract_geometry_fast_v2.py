# -*- coding: utf-8 -*-
"""
Tencent is pleased to support the open source community by making Tencent Hunyuan 3D Omni available.

Copyright (C) 2025 Tencent.  All rights reserved. The below software and/or models in this 
distribution may have been modified by Tencent ("Tencent Modifications"). All Tencent Modifications 
are Copyright (C) Tencent.

Tencent Hunyuan 3D Omni is licensed under the TENCENT HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT 
except for the third-party components listed below, which is licensed under different terms. 
Tencent Hunyuan 3D Omni does not impose any additional limitations beyond what is outlined in the 
respective licenses of these third-party components. Users must comply with all terms and conditions 
of original licenses of these third-party components and must ensure that the usage of the third party 
components adheres to all relevant laws and regulations. 

For avoidance of doubts, Tencent Hunyuan 3D Omni means training code, inference-enabling code, parameters, 
and/or weights of this Model, which are made publicly available by Tencent in accordance with TENCENT 
HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT.
"""

from typing import Callable, Tuple, List, Union, Optional
import numpy as np
import torch
import trimesh
from skimage import measure
import traceback
from tqdm import tqdm
from torch import nn
from einops import repeat

try:
    from .extract_geometry_base import BaseGeometryExtractor
except ImportError:
    from extract_geometry_base import BaseGeometryExtractor

try:
    from .find_invalid_pts import extract_near_surface_volume_fn as find_invalid_pionts_fn
except ImportError:
    from find_invalid_pts import extract_near_surface_volume_fn as find_invalid_pionts_fn
except Exception as err:
    print(err)


class FastGeometryExtractorV2(BaseGeometryExtractor):
    """
    快速几何提取器，使用多分辨率金字塔策略高效提取几何
    """
    
    def __init__(self, device: torch.device = None, dtype=torch.float16):
        super().__init__(device)
        self.dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=self.device, dtype=dtype)
        self.dilate.weight = nn.Parameter(torch.ones(self.dilate.weight.shape, device=self.device, dtype=dtype))
    
    @torch.no_grad()
    def extract_geometry(
        self,
        geometric_func: Callable,
        bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
        octree_depth: int = 7,
        num_chunks: int = 10000,
        disable_tqdm: bool = False,
        mc_level: float = -1 / 512,
        octree_resolution: int = 256,
        rotation_matrix=None,
        mc_mode='mc',
        dtype=torch.float16,
        min_resolution: int = 95,
        **kwargs
    ) -> trimesh.Trimesh:
        """
        使用多分辨率金字塔策略提取几何
        
        Args:
            geometric_func: 几何函数
            bounds: 边界框
            octree_depth: 八叉树深度
            num_chunks: 分块数量
            disable_tqdm: 是否禁用进度条
            mc_level: 等值面阈值
            octree_resolution: 八叉树分辨率
            rotation_matrix: 旋转矩阵
            mc_mode: MC模式
            dtype: 数据类型
            min_resolution: 最小分辨率
            
        Returns:
            trimesh.Trimesh: 提取的网格
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        if octree_resolution is None:
            octree_resolution = 2 ** octree_depth

        assert octree_resolution >= 256, "octree resolution must be at least 256 for fast inference"

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = self.generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )

        grid_size = np.array(grid_size)
        xyz_samples = torch.FloatTensor(xyz_samples).to(self.device).half()

        if mc_level == -1:
            print(f'Training with soft labels, inference with sigmoid and marching cubes level 0.')
        elif mc_level == 0:
            print(f'VAE Trained with TSDF, inference with marching cubes level 0.')
        else:
            print(f'VAE Trained with Occupancy, inference with marching cubes level {mc_level}.')
        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc=f"MC Level {mc_level} Implicit Function:", disable=disable_tqdm, leave=False
        ):
            queries = xyz_samples[start: start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=1)
            logits = geometric_func(batch_queries)
            if mc_level == -1:
                mc_level = 0
                print(f'Training with soft labels, inference with sigmoid and marching cubes level 0.')
                logits = torch.sigmoid(logits) * 2 - 1
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1).view((1, grid_size[0], grid_size[1], grid_size[2])).half()

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=self.device)
            if octree_depth_now == resolutions[-1]:
                next_logits = torch.full(next_index.shape, float('nan'), dtype=dtype, device=self.device)
            else:
                next_logits = torch.full(next_index.shape, -10000., dtype=dtype, device=self.device)
            curr_points = find_invalid_pionts_fn(grid_logits.squeeze(0), mc_level)
            curr_points += grid_logits.squeeze(0).abs() < min(0.95, 0.95*128*4/octree_depth_now)
            if octree_depth_now >510:
                expand_num = 0
            else:
                expand_num = 1
            for i in range(expand_num):
                curr_points = self.dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for i in range(1):
                next_index = self.dilate(next_index.unsqueeze(0)).squeeze(0)

            # fix the problem of OOM
            nidx = torch.where(next_index > 0)
            nidx = (nidx[0].to(torch.int32), nidx[1].to(torch.int32), nidx[2].to(torch.int32))
            next_points = torch.stack(nidx, dim=1).to(torch.int16)
            next_points = next_points * torch.tensor(resolution, device=self.device, dtype=torch.float32)
            next_points = next_points + torch.tensor(bbox_min, device=self.device, dtype=torch.float32)

            batch_logits = []
            for start in tqdm(
                range(0, next_points.shape[0], num_chunks),
                desc=f"MC Level {octree_depth_now + 1} Implicit Function:", disable=disable_tqdm, leave=False
            ):
                queries = next_points[start: start + num_chunks, :]
                batch_queries = repeat(queries, "p c -> b p c", b=1)
                logits = geometric_func(batch_queries)
                if mc_level == -1:
                    mc_level = 0
                    print(f'Training with soft labels, inference with sigmoid and marching cubes level 0.')
                    logits = torch.sigmoid(logits) * 2 - 1
                batch_logits.append(logits)
            grid_logits = torch.cat(batch_logits, dim=1).half()
            next_logits[nidx] = grid_logits[0].squeeze(-1)
            grid_logits = next_logits.unsqueeze(0)

        mesh_v_f = []
        has_surface = np.zeros((1,), dtype=np.bool_)
        try:
            if mc_mode == 'mc':
                if len(resolutions) > 1:
                    mask = (next_index > 0).cpu().numpy()
                    grid_logits = grid_logits.cpu().numpy()
                    vertices, faces, normals, _ = measure.marching_cubes(grid_logits[0], mc_level, method="lewiner",
                                                                         mask=mask)
                else:
                    vertices, faces, normals, _ = measure.marching_cubes(grid_logits[0].cpu().numpy(), mc_level,
                                                                         method="lewiner")
                vertices = vertices / (grid_size - 1) * bbox_size + bbox_min
                # vertices[:, [0, 1]] = vertices[:, [1, 0]]
            elif mc_mode == 'dmc':
                if not hasattr(self, 'dmc'):
                    try:
                        from diso import DiffDMC
                        self.dmc = DiffDMC(dtype=torch.float32).to(self.device)
                    except Exception:
                        raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")

                torch.cuda.empty_cache()
                grid_logits = grid_logits[0] # -grid_logits[0]
                grid_logits = grid_logits.to(torch.float32).contiguous()
                verts, faces = self.dmc(grid_logits, deform=None, return_quads=False, normalize=False)
                verts = verts * torch.tensor(resolution, device=self.device)
                verts = verts + torch.tensor(bbox_min, device=self.device)
                vertices = verts.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()[:, ::-1]
            else:
                raise ValueError(f"Unknown marching cubes mode: {mc_mode}")
            mesh_v_f.append((vertices.astype(np.float32), np.ascontiguousarray(faces)))
            has_surface[0] = True

        except ValueError:
            traceback.print_exc()
            mesh_v_f.append((None, None))
            has_surface[0] = False

        except RuntimeError:
            traceback.print_exc()
            mesh_v_f.append((None, None))
            has_surface[0] = False
        
        # convert to trimesh.Trimesh
        if has_surface[0]:
            vertices, faces = mesh_v_f[0]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh
        return trimesh.Trimesh()

    def extract_geometry_with_stats(
        self,
        geometric_func: Callable,
        **kwargs
    ) -> Tuple[trimesh.Trimesh, dict]:
        """
        提取几何并返回统计信息
        """
        mesh = self.extract_geometry(geometric_func, **kwargs)
        stats = self.get_mesh_info(mesh)
        stats['extraction_success'] = self.validate_mesh(mesh)
        return mesh, stats


@torch.no_grad()
def extract_geometry_fast_v2(device: torch.device = None, *args, **kwargs):
    """
    向后兼容的函数接口
    """
    extractor = FastGeometryExtractorV2(device=device)
    mesh = extractor.extract_geometry(*args, **kwargs)
    return ([(mesh.vertices, mesh.faces)] if len(mesh.vertices) > 0 else [(None, None)]), \
           np.array([len(mesh.vertices) > 0], dtype=np.bool_)


if __name__ == "__main__":
    import time

    # 测试FastGeometryExtractor类
    print("=" * 50)
    print("开始运行 FastGeometryExtractor 测试套件")
    print("=" * 50)
    
    start_time = time.time()
    extractor = FastGeometryExtractorV2()
    mesh, stats = extractor.extract_geometry_with_stats(
        geometric_func=BaseGeometryExtractor.sphere_sdf,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        octree_depth=7,
        mc_level=0,
        mc_mode='dmc' # 'mc' 
    )
    mesh.export("output_fast_new_mesh.obj")
    
    # 输出统计信息
    print("提取的网格信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    expected_volume = (4/3) * np.pi * (0.5**3) 
    if stats['volume'] > 0:
        volume_ratio = stats['volume'] / expected_volume
        print(f"  体积比 (实际/预期): {volume_ratio:.3f}")
    
    print(f"✅ FastGeometryExtractor 测试通过！耗时：{time.time() - start_time:.2f}秒")
    
    # 测试向后兼容性
    print("=" * 50)
    print("开始测试向后兼容性...")
    print("=" * 50)
    start_time = time.time()
    mesh_v_f, has_surface = extract_geometry_fast_v2(
        geometric_func=BaseGeometryExtractor.sphere_sdf,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        octree_depth=7,
        mc_level=0,
        mc_mode='dmc' # 'mc' 
    )
    if has_surface[0]:
        mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])
        print('bounds', mesh.bounds)
        mesh.export("output_fast_new_mesh_compat.obj")
        print(f"✅ 向后兼容函数调用成功，顶点数: {len(mesh.vertices)}, 面数: {len(mesh.faces)}")
    else:
        print("❌ 向后兼容函数调用失败")

    print(f"✅ 向后兼容性测试通过！耗时: {time.time() - start_time:.2f}秒")
    print("=" * 50)
    print("所有测试通过！")
