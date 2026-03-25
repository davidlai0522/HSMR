from lib.kits.hsmr_demo import *
import json
import trimesh
import argparse

# ================== Helper Functions ==================

def extract_bone_scaling(pipeline, pd_params):
    """Extract bone scaling data from SKEL model"""
    B = 720
    bone_scales_all = []
    joints_all = []
    
    for bw in asb(total=len(pd_params['poses']), bs_scope=B, enable_tqdm=False):
        poses_batch = pd_params['poses'][bw.sid:bw.eid].to(pipeline.device)
        betas_batch = pd_params['betas'][bw.sid:bw.eid].to(pipeline.device)
        
        batch_size = len(poses_batch)
        
        # Get shaped vertices using einsum
        skin_v0 = pipeline.skel_model.skin_template_v
        shapedirs = pipeline.skel_model.shapedirs
        
        # shapedirs: (num_verts, 3, num_betas), betas: (batch_size, num_betas)
        # Result: (batch_size, num_verts, 3)
        v_shaped = skin_v0 + torch.einsum('ijk,bk->bij', shapedirs, betas_batch)
        
        # Get joint locations (absolute positions)
        J = torch.einsum('bik,ji->bjk', [v_shaped, pipeline.skel_model.J_regressor_osim])
        
        # Compute relative joint positions (bone vectors)
        # J_[0] = J[0] (root stays as absolute position)
        # J_[i] = J[i] - J[parent[i]] for i > 0
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, pipeline.skel_model.parent, :]
        
        # Compute bone scaling
        bone_scale = pipeline.skel_model.compute_bone_scale(
            J_, v_shaped, skin_v0.unsqueeze(0).expand(batch_size, -1, -1), 
            is_unique_beta=False
        )
        
        bone_scales_all.append(bone_scale.detach().cpu())
        joints_all.append(J.detach().cpu())  # Store absolute positions for reference
    
    bone_scales = torch.cat(bone_scales_all, dim=0)  # (N, num_joints, 3)
    joints = torch.cat(joints_all, dim=0)  # (N, num_joints, 3)
    
    return {
        'bone_scales': bone_scales,
        'joints': joints,
        'joint_names': pipeline.skel_model.joints_name,
        'bone_names': pipeline.skel_model.bone_names
    }

# ================== Command Line Supports ==================
def load_img_inputs(args, MAX_IMG_W=1920, MAX_IMG_H=1080):
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

    # 1. Inference inputs type.
    inputs_path = Path(args.input_path)
    if inputs_path.is_file() and inputs_path.suffix.lower() in IMG_EXTS:
        inputs_type = 'imgs'
        single_img = True
    elif inputs_path.is_file():
        inputs_type = 'video'
        single_img = False
    else:
        inputs_type = 'imgs'
        single_img = False
    get_logger(brief=True).info(f'🚚 Loading inputs from: {inputs_path}, regarded as <{inputs_type}>.')

    # 2. Load inputs.
    inputs_meta = {'type': inputs_type}
    if inputs_type == 'video':
        inputs_meta['seq_name'] = inputs_path.stem
        frames, _ = load_video(inputs_path)
        if frames.shape[1] > MAX_IMG_H:
            frames = flex_resize_video(frames, (MAX_IMG_H, -1), kp_mod=4)
        if frames.shape[2] > MAX_IMG_W:
            frames = flex_resize_video(frames, (-1, MAX_IMG_W), kp_mod=4)
        raw_imgs = [frame for frame in frames]
    elif inputs_type == 'imgs':
        if single_img:
            img_fns = [inputs_path]
            inputs_meta['seq_name'] = inputs_path.stem
        else:
            img_fns = list(inputs_path.glob('*.*'))
            img_fns = [fn for fn in img_fns if fn.suffix.lower() in IMG_EXTS]
            inputs_meta['seq_name'] = f'{inputs_path.stem}-img_cnt={len(img_fns)}'
        raw_imgs = []
        for fn in img_fns:
            img, _ = load_img(fn)
            if img.shape[0] > MAX_IMG_H:
                img = flex_resize_img(img, (MAX_IMG_H, -1), kp_mod=4)
            if img.shape[1] > MAX_IMG_W:
                img = flex_resize_img(img, (-1, MAX_IMG_W), kp_mod=4)
            raw_imgs.append(img)
        inputs_meta['img_fns'] = img_fns
    else:
        raise ValueError(f'Unsupported inputs type: {inputs_type}.')
    get_logger(brief=True).info(f'📦 Totally {len(raw_imgs)} images are loaded.')

    return raw_imgs, inputs_meta

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--input_type', type=str, default='auto', help='Specify the input type. auto: file~video, folder~imgs', choices=['auto', 'video', 'imgs'])
    parser.add_argument('-i', '--input_path', type=str, required=True, help='The input images root or video file path.')
    parser.add_argument('-o', '--output_path', type=str, default=PM.outputs/'demos', help='The output root.')
    parser.add_argument('-m', '--model_root', type=str, default=DEFAULT_HSMR_ROOT, help='The model root which contains `.hydra/config.yaml`.')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='The device.')
    parser.add_argument('--det_bs', type=int, default=10, help='The max batch size for detector.')
    parser.add_argument('--det_mis', type=int, default=512, help='The max image size for detector.')
    parser.add_argument('--rec_bs', type=int, default=300, help='The batch size for recovery.')
    parser.add_argument('--max_instances', type=int, default=5, help='Max instances activated in one image.')
    parser.add_argument('--ignore_skel', action='store_true', help='Do not render skeleton to boost the rendering.')
    parser.add_argument('--have_caption', action='store_true', help='Add caption to the rendered images.')
    parser.add_argument('--save_mesh', action='store_true', help='Save skeleton and skin meshes as OBJ files.')
    parser.add_argument('--save_json', action='store_true', default=True, help='Save SKEL data as JSON files.')
    parser.add_argument('--save_viz', action='store_true', help='Save visualization images/videos like run_demo.')
    parser.add_argument('--mesh_scale', type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=('X', 'Y', 'Z'),
                        help='Manual scaling factors for mesh output [scale_x, scale_y, scale_z]. Default: 1.0 1.0 1.0')
    args = parser.parse_args()
    return args

def main():
    # ⛩️ 0. Preparation.
    args = parse_args()
    outputs_root = Path(args.output_path)
    outputs_root.mkdir(parents=True, exist_ok=True)

    monitor = TimeMonitor()

    # ⛩️ 1. Preprocess.

    with monitor('Data Preprocessing'):
        with monitor('Load Inputs'):
            raw_imgs, inputs_meta = load_img_inputs(args)

        with monitor('Detector Initialization'):
            get_logger(brief=True).info('🧱 Building detector.')
            detector = build_detector(
                    batch_size   = args.det_bs,
                    max_img_size = args.det_mis,
                    device       = args.device,
                )

        with monitor('Detecting'):
            get_logger(brief=True).info(f'🖼️ Detecting...')
            detector_outputs = detector(raw_imgs)

        with monitor('Patching & Loading'):
            patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, args.max_instances)  # N * (256, 256, 3)
        if len(patches) == 0:
            get_logger(brief=True).error(f'🚫 No human instance detected. Please ensure the validity of your inputs!')
        get_logger(brief=True).info(f'🔍 Totally {len(patches)} human instances are detected.')

    # Input should be image only (as this script is designed to run on cpu)
    inputs_meta['type'] == 'imgs'

    # ⛩️ 2. Human skeleton and mesh recovery.
    with monitor('Pipeline Initialization'):
        get_logger(brief=True).info(f'🧱 Building recovery pipeline.')
        pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)

    with monitor('Recovery'):
        get_logger(brief=True).info(f'🏃 Recovering with B={args.rec_bs}...')
        pd_params, pd_cam_t = [], []
        for bw in asb(total=len(patches), bs_scope=args.rec_bs, enable_tqdm=True):
            patches_i = patches[bw.sid:bw.eid]  # (N, 256, 256, 3)
            patches_normalized_i = (patches_i - IMG_MEAN_255) / IMG_STD_255  # (N, 256, 256, 3)
            patches_normalized_i = patches_normalized_i.transpose(0, 3, 1, 2)  # (N, 3, 256, 256)
            with torch.no_grad():
                outputs = pipeline(patches_normalized_i)
            pd_params.append({k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()})
            pd_cam_t.append(outputs['pd_cam_t'].detach().cpu().clone())

        pd_params = assemble_dict(pd_params, expand_dim=False)  # [{k:[x]}, {k:[y]}] -> {k:[x, y]}
        pd_cam_t = torch.cat(pd_cam_t, dim=0)
        dump_data = {
                'patch_cam_t' : pd_cam_t.numpy(),
                **{k: v.numpy() for k, v in pd_params.items()},
            }

        get_logger(brief=True).info(f'🤌 Preparing meshes...')
        m_skin, m_skel = prepare_mesh(pipeline, pd_params)
        
        # Apply manual scaling to mesh vertices and camera translation
        if args.mesh_scale != [1.0, 1.0, 1.0]:
            scale_factors = torch.tensor(args.mesh_scale, dtype=m_skin['v'].dtype, device=m_skin['v'].device)
            m_skin['v'] = m_skin['v'] * scale_factors
            m_skel['v'] = m_skel['v'] * scale_factors
            
            # Scale camera translation to maintain alignment
            # pd_cam_t has shape (N, 3) with [tx, ty, tz]
            cam_scale = torch.tensor(args.mesh_scale, dtype=pd_cam_t.dtype, device=pd_cam_t.device)
            pd_cam_t = pd_cam_t * cam_scale
            
            get_logger(brief=True).info(f'📐 Applied manual scaling: {args.mesh_scale}')
        
        get_logger(brief=True).info(f'🏁 Done.')
        
        # Extract bone scaling data
        get_logger(brief=True).info(f'📏 Extracting bone scaling data...')
        bone_data = extract_bone_scaling(pipeline, pd_params)
        get_logger(brief=True).info(f'✅ Bone scaling extracted.')

    # ⛩️ 3. Visualization (optional)
    if args.save_viz:
        with monitor('Visualization'):
            if args.ignore_skel:
                m_skel_viz = None
            else:
                m_skel_viz = m_skel
            results, full_cam_t = visualize_full_img(pd_cam_t, raw_imgs, det_meta, m_skin, m_skel_viz, args.have_caption)
            dump_data['full_cam_t'] = full_cam_t
            # Save rendering and dump results.
            if inputs_meta['type'] == 'video':
                seq_name = f'{pipeline.name}-' + inputs_meta['seq_name']
                save_video(results, outputs_root / f'{seq_name}.mp4')
                # Dump data for each frame, here `i` refers to frames, `j` refers to image patches.
                dump_results = []
                cur_patch_j = 0
                for i in range(len(raw_imgs)):
                    n_patch_cur_img = det_meta['n_patch_per_img'][i]
                    dump_results_i = {k: v[cur_patch_j:cur_patch_j+n_patch_cur_img] for k, v in dump_data.items()}
                    dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
                    cur_patch_j += n_patch_cur_img
                    dump_results.append(dump_results_i)
                np.save(outputs_root / f'{seq_name}.npy', dump_results)
            elif inputs_meta['type'] == 'imgs':
                img_names = [f'{pipeline.name}-{fn.name}' for fn in inputs_meta['img_fns']]
                # Dump data for each image separately, here `i` refers to images, `j` refers to image patches.
                cur_patch_j = 0
                for i, img_name in enumerate(tqdm(img_names, desc='Saving images')):
                    n_patch_cur_img = det_meta['n_patch_per_img'][i]
                    dump_results_i = {k: v[cur_patch_j:cur_patch_j+n_patch_cur_img] for k, v in dump_data.items()}
                    dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
                    cur_patch_j += n_patch_cur_img
                    save_img(results[i], outputs_root / f'{img_name}.jpg')
                    np.savez(outputs_root / f'{img_name}.npz', **dump_results_i)

            get_logger(brief=True).info(f'🎨 Rendering results are under {outputs_root}.')

    # ⛩️ 4. Extract and save SKEL data to JSON/Mesh
    if args.save_json or args.save_mesh:
        with monitor('Saving SKEL Data'):
            get_logger(brief=True).info(f'💾 Extracting SKEL data...')
            
            # Prepare SKEL data for each detected instance
            cur_patch_j = 0
            for i in range(len(raw_imgs)):
                n_patch_cur_img = det_meta['n_patch_per_img'][i]
                
                for j in range(n_patch_cur_img):
                    instance_idx = cur_patch_j + j
                    
                    # Determine filenames
                    if inputs_meta['type'] == 'imgs':
                        img_name = inputs_meta['img_fns'][i].stem
                        json_filename = outputs_root / f'{pipeline.name}-{img_name}_instance_{j}.json'
                        mesh_prefix = outputs_root / f'{pipeline.name}-{img_name}_instance_{j}'
                    elif inputs_meta['type'] == 'video':
                        seq_name = inputs_meta['seq_name']
                        json_filename = outputs_root / f'{pipeline.name}-{seq_name}_frame_{i}_instance_{j}.json'
                        mesh_prefix = outputs_root / f'{pipeline.name}-{seq_name}_frame_{i}_instance_{j}'
                    
                    saved_files = []
                    
                    # Save JSON file if requested
                    if args.save_json:
                        # Extract SKEL data for this instance
                        skel_data = {
                            'image_index': i,
                            'instance_index': instance_idx,
                            'instance_in_image': j,
                            
                            # # SKEL mesh data
                            # 'skel_vertices': m_skel['v'][instance_idx].numpy().tolist(),  # (Ve, 3)
                            # 'skel_faces': m_skel['f'].cpu().numpy().tolist(),  # (F, 3)
                            
                            # # Skin mesh data
                            # 'skin_vertices': m_skin['v'][instance_idx].numpy().tolist(),  # (Vi, 3)
                            # 'skin_faces': m_skin['f'].cpu().numpy().tolist(),  # (F, 3)
                            
                            # Camera and bounding box info
                            'patch_cam_t': pd_cam_t[instance_idx].numpy().tolist(),  # (3,)
                            'bbx_cs': det_meta['bbx_cs_per_img'][i][j].tolist() if j < len(det_meta['bbx_cs_per_img'][i]) else None,
                            
                            # SKEL parameters
                            'poses': pd_params['poses'][instance_idx].numpy().tolist(),
                            'betas': pd_params['betas'][instance_idx].numpy().tolist(),
                            
                            # Bone scaling data
                            'bone_scales': bone_data['bone_scales'][instance_idx].numpy().tolist(),  # (num_joints, 3)
                            'joints': bone_data['joints'][instance_idx].numpy().tolist(),  # (num_joints, 3)
                            'joint_names': bone_data['joint_names'],
                            'bone_names': bone_data['bone_names'],
                        }
                        
                        # Add any additional parameters from pd_params
                        for key, value in pd_params.items():
                            if key not in ['poses', 'betas']:
                                skel_data[key] = value[instance_idx].numpy().tolist()
                        
                        with open(json_filename, 'w') as f:
                            json.dump(skel_data, f, indent=2)
                        saved_files.append(json_filename.name)
                    
                    # Save mesh files if requested
                    if args.save_mesh:
                        # Save skeleton mesh as OBJ file
                        skel_mesh = trimesh.Trimesh(
                            vertices=m_skel['v'][instance_idx].numpy(),
                            faces=m_skel['f'].cpu().numpy(),
                        )
                        skel_mesh_filename = f'{mesh_prefix}_skeleton.obj'
                        skel_mesh.export(skel_mesh_filename, file_type='obj')
                        saved_files.append(f'{mesh_prefix.name}_skeleton.obj')
                        
                        # Save skin mesh as OBJ file
                        skin_mesh = trimesh.Trimesh(
                            vertices=m_skin['v'][instance_idx].numpy(),
                            faces=m_skin['f'].cpu().numpy(),
                        )
                        skin_mesh_filename = f'{mesh_prefix}_skin.obj'
                        skin_mesh.export(skin_mesh_filename, file_type='obj')
                        saved_files.append(f'{mesh_prefix.name}_skin.obj')
                    
                    if saved_files:
                        get_logger(brief=True).info(f'💾 Saved: {", ".join(saved_files)}')
                
                cur_patch_j += n_patch_cur_img
            
            # Also save a summary file with all instances if saving JSON
            if args.save_json:
                summary_data = {
                    'metadata': {
                        'input_type': inputs_meta['type'],
                        'num_images': len(raw_imgs),
                        'total_instances': len(patches),
                        'model_name': pipeline.name,
                    },
                    'instances': []
                }
                
                cur_patch_j = 0
                for i in range(len(raw_imgs)):
                    n_patch_cur_img = det_meta['n_patch_per_img'][i]
                    for j in range(n_patch_cur_img):
                        instance_idx = cur_patch_j + j
                        summary_data['instances'].append({
                            'image_index': i,
                            'instance_index': instance_idx,
                            'instance_in_image': j,
                            'num_skel_vertices': len(m_skel['v'][instance_idx]),
                            'num_skin_vertices': len(m_skin['v'][instance_idx]),
                            'num_skel_faces': len(m_skel['f']),
                            'num_skin_faces': len(m_skin['f']),
                        })
                    cur_patch_j += n_patch_cur_img
                
                summary_filename = outputs_root / f'{pipeline.name}-summary.json'
                with open(summary_filename, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                
                get_logger(brief=True).info(f'📊 Saved summary: {summary_filename.name}')
            
            get_logger(brief=True).info('🎊 All SKEL data saved to')
            get_logger(brief=True).info(f"\033[92m{outputs_root}\033[0m")
    else:
        get_logger(brief=True).info(f'⏭️ Skipping data export (use --save_json and/or --save_mesh to enable)')
    
    get_logger(brief=True).info(f'✨ Everything is done!')
    monitor.report()


if __name__ == '__main__':
    main()