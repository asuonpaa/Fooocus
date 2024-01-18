import comfy.utils
import folder_paths
from modules.config import path_checkpoints, path_loras
import copy

folder_paths.folder_names_and_paths["fooocus_checkpoints"] = ([path_checkpoints], {".safetensors", ".ckpt"})
folder_paths.folder_names_and_paths["fooocus_loras"] = ([path_loras], {".safetensors", ".ckpt"})

class AsyncTask:
    def __init__(self, args):
        self.args = args
        self.yields = []
        self.results = []

def refresh_pipeline(pipeline, p):
    import extras.ip_adapter as ip_adapter
    print("Refreshing Fooocus pipeline")
    pipeline.refresh_everything(refiner_model_name=p["refiner"], base_model_name=p["base_model"],
            loras=p["loras"], base_model_additional_loras=p["additional_loras"], use_synthetic_refiner=p["use_synthetic_refiner"])
    if p["ip_tasks"]:
        print(f"Patching unet with {len(p['ip_tasks'])} ip-adapter tasks")
        pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, p["ip_tasks"])


# TODO Remove
def progressbar(async_task, number, text):
    print(f'[Fooocus] {text}')

def processTaskSimple(async_task, pipeline_in, positive_cond, negative_cond, seed, vary_image):
    # TODO Remove unused
    import os
    import traceback
    import math
    import numpy as np
    import torch
    import time
    import shared
    import random
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.config
    import modules.patch
    import ldm_patched.modules.model_management
    import extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import modules.advanced_parameters as advanced_parameters
    import extras.ip_adapter as ip_adapter
    import extras.face_crop
    import fooocus_version

    from modules.sdxl_styles import apply_style, apply_wildcards, fooocus_expansion
    from modules.private_logger import log
    from extras.expansion import safe_str
    from modules.util import remove_empty_str, HWC3, resize_image, \
        get_image_shape_ceil, set_image_shape_ceil, get_shape_ceil, resample_image, erode_or_dilate

    execution_start_time = time.perf_counter()

    args = async_task.args
    args.reverse()

    performance_selection = args.pop()
    aspect_ratios_selection = args.pop()
    image_number = args.pop()
    sharpness = args.pop()
    guidance_scale = args.pop()
#    base_model_name = args.pop()
    refiner_model_name = args.pop()
    refiner_switch = args.pop()
    loras = [[str(args.pop()), float(args.pop())] for _ in range(5)]
    uov_method = args.pop()
    outpaint_selections = args.pop()
    inpaint_input_image = args.pop()
    inpaint_additional_prompt = args.pop()
    inpaint_mask_image_upload = args.pop()

    if vary_image is not None:
        uov_input_image = np.asarray(core.pytorch_to_numpy(vary_image[0]))
    else:
        uov_input_image = None

#    cn_tasks = {x: [] for x in flags.ip_list}
#    for _ in range(4):
#        cn_img = args.pop()
#        cn_stop = args.pop()
#        cn_weight = args.pop()
#        cn_type = args.pop()
#        if cn_img is not None:
#            cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

    outpaint_selections = [o.lower() for o in outpaint_selections]
    base_model_additional_loras = []
    uov_method = uov_method.lower()

#    if base_model_name == refiner_model_name:
#        print(f'Refiner disabled because base model and refiner are same.')
#        refiner_model_name = 'None'

    assert performance_selection in ['Speed', 'Quality', 'Extreme Speed']

    steps = 30

    if performance_selection == 'Speed':
        steps = 30

    if performance_selection == 'Quality':
        steps = 60

    if performance_selection == 'Extreme Speed':
        print('Enter LCM mode.')
        progressbar(async_task, 1, 'Downloading LCM components ...')
        loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]

        if refiner_model_name != 'None':
            print(f'Refiner disabled in LCM mode.')

        refiner_model_name = 'None'
        sampler_name = advanced_parameters.sampler_name = 'lcm'
        scheduler_name = advanced_parameters.scheduler_name = 'lcm'
        modules.patch.sharpness = sharpness = 0.0
        cfg_scale = guidance_scale = 1.0
        modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
        refiner_switch = 1.0
        modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
        modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
        modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0
        steps = 8

    modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg
    print(f'[Parameters] Adaptive CFG = {modules.patch.adaptive_cfg}')

    modules.patch.sharpness = sharpness
    print(f'[Parameters] Sharpness = {modules.patch.sharpness}')

    modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive
    modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative
    modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end
    print(f'[Parameters] ADM Scale = '
          f'{modules.patch.positive_adm_scale} : '
          f'{modules.patch.negative_adm_scale} : '
          f'{modules.patch.adm_scaler_end}')

    cfg_scale = float(guidance_scale)
    print(f'[Parameters] CFG = {cfg_scale}')

    initial_latent = None
    denoising_strength = 1.0
    tiled = False

    width, height = aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
    width, height = int(width), int(height)

    refiner_swap_method = advanced_parameters.refiner_swap_method

    inpaint_worker.current_task = None
    inpaint_parameterized = advanced_parameters.inpaint_engine != 'None'
    inpaint_image = None
    inpaint_mask = None
    inpaint_head_model_path = None

    use_synthetic_refiner = False

#    controlnet_canny_path = None
#    controlnet_cpds_path = None
#    clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

    print(f'[Parameters] Seed = {seed}')

    sampler_name = advanced_parameters.sampler_name
    scheduler_name = advanced_parameters.scheduler_name

    goals = []
    tasks = []

    if uov_input_image is not None:
        uov_input_image = HWC3(uov_input_image)
        goals.append('vary')

#    if isinstance(inpaint_input_image, dict):
#        inpaint_image = inpaint_input_image['image']
#        inpaint_mask = inpaint_input_image['mask'][:, :, 0]
#
#        if advanced_parameters.inpaint_mask_upload_checkbox:
#            if isinstance(inpaint_mask_image_upload, np.ndarray):
#                if inpaint_mask_image_upload.ndim == 3:
#                    H, W, C = inpaint_image.shape
#                    inpaint_mask_image_upload = resample_image(inpaint_mask_image_upload, width=W, height=H)
#                    inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
#                    inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
#                    inpaint_mask = np.maximum(inpaint_mask, inpaint_mask_image_upload)
#
#        if int(advanced_parameters.inpaint_erode_or_dilate) != 0:
#            inpaint_mask = erode_or_dilate(inpaint_mask, advanced_parameters.inpaint_erode_or_dilate)
#
#        if advanced_parameters.invert_mask_checkbox:
#            inpaint_mask = 255 - inpaint_mask
#
#        inpaint_image = HWC3(inpaint_image)
#        if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
#                and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
#            progressbar(async_task, 1, 'Downloading upscale models ...')
#            modules.config.downloading_upscale_model()
#            if inpaint_parameterized:
#                progressbar(async_task, 1, 'Downloading inpainter ...')
#                inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
#                    advanced_parameters.inpaint_engine)
#                base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
#                print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
#                if refiner_model_name == 'None':
#                    use_synthetic_refiner = True
#                    refiner_switch = 0.5
#            else:
#                inpaint_head_model_path, inpaint_patch_model_path = None, None
#                print(f'[Inpaint] Parameterized inpaint is disabled.')
#            if inpaint_additional_prompt != '':
#                if prompt == '':
#                    prompt = inpaint_additional_prompt
#                else:
#                    prompt = inpaint_additional_prompt + '\n' + prompt
#            goals.append('inpaint')

#    if current_tab == 'ip':
#        goals.append('cn')
#        progressbar(async_task, 1, 'Downloading control models ...')
#        if len(cn_tasks[flags.cn_canny]) > 0:
#            controlnet_canny_path = modules.config.downloading_controlnet_canny()
#        if len(cn_tasks[flags.cn_cpds]) > 0:
#            controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
#        if len(cn_tasks[flags.cn_ip]) > 0:
#            clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
#        if len(cn_tasks[flags.cn_ip_face]) > 0:
#            clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
#                'face')

#    # Load or unload CNs
#    pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
#    ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
#    ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

    # TODO Ari: Remove refiner?
    switch = int(round(steps * refiner_switch))

    print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
    print(f'[Parameters] Steps = {steps} - {switch}')

#    pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
#                                loras=loras, base_model_additional_loras=base_model_additional_loras,
#                                use_synthetic_refiner=use_synthetic_refiner)

    # TODO: Remove
#    pipeline.refresh_base_model("realisticStockPhoto_v10.safetensors")
#    pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name="juggernautXL_v8Rundiffusion.safetensors",
#                                loras=loras, base_model_additional_loras=base_model_additional_loras,
#                                use_synthetic_refiner=use_synthetic_refiner)

#    pipeline.refresh_loras(loras=loras, base_model_additional_loras=base_model_additional_loras)
#    pipeline_in["loras"] = loras
#    pipeline_in["additional_loras"] = base_model_additional_loras




    refresh_pipeline(pipeline, pipeline_in)




#    pipeline.refresh_loras(loras=loras, base_model_additional_loras=base_model_additional_loras)

    # ----------------------------------------------------------------------------------------

    if len(goals) > 0:
        progressbar(async_task, 13, 'Image processing ...')

    if 'vary' in goals:
        if 'subtle' in uov_method:
            denoising_strength = 0.5
        if 'strong' in uov_method:
            denoising_strength = 0.85

#        shape_ceil = get_image_shape_ceil(uov_input_image)
#        if shape_ceil < 1024:
#            print(f'[Vary] Image is resized because it is too small.')
#            shape_ceil = 1024
#        elif shape_ceil > 2048:
#            print(f'[Vary] Image is resized because it is too big.')
#            shape_ceil = 2048
        shape_ceil = 1024

        uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)

        initial_pixels = core.numpy_to_pytorch(uov_input_image)
        progressbar(async_task, 13, 'VAE encoding ...')

        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=steps,
            switch=switch,
            denoise=denoising_strength,
            refiner_swap_method=refiner_swap_method
        )

        initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
        B, C, H, W = initial_latent['samples'].shape
        width = W * 8
        height = H * 8
        print(f'Final resolution is {str((height, width))}.')

#    if 'inpaint' in goals:
#        if len(outpaint_selections) > 0:
#            H, W, C = inpaint_image.shape
#            if 'top' in outpaint_selections:
#                inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
#                inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
#                                      constant_values=255)
#            if 'bottom' in outpaint_selections:
#                inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
#                inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
#                                      constant_values=255)
#
#            H, W, C = inpaint_image.shape
#            if 'left' in outpaint_selections:
#                inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
#                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant',
#                                      constant_values=255)
#            if 'right' in outpaint_selections:
#                inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
#                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant',
#                                      constant_values=255)
#
#            inpaint_image = np.ascontiguousarray(inpaint_image.copy())
#            inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
#            advanced_parameters.inpaint_strength = 1.0
#            advanced_parameters.inpaint_respective_field = 1.0
#
#        denoising_strength = advanced_parameters.inpaint_strength
#
#        inpaint_worker.current_task = inpaint_worker.InpaintWorker(
#            image=inpaint_image,
#            mask=inpaint_mask,
#            use_fill=denoising_strength > 0.99,
#            k=advanced_parameters.inpaint_respective_field
#        )
#
#        if advanced_parameters.debugging_inpaint_preprocessor:
#            yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(),
#                         do_not_show_finished_images=True)
#            return
#
#        progressbar(async_task, 13, 'VAE Inpaint encoding ...')
#
#        inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
#        inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
#        inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)
#
#        candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
#            steps=steps,
#            switch=switch,
#            denoise=denoising_strength,
#            refiner_swap_method=refiner_swap_method
#        )
#
#        latent_inpaint, latent_mask = core.encode_vae_inpaint(
#            mask=inpaint_pixel_mask,
#            vae=candidate_vae,
#            pixels=inpaint_pixel_image)
#
#        latent_swap = None
#        if candidate_vae_swap is not None:
#            progressbar(async_task, 13, 'VAE SD15 encoding ...')
#            latent_swap = core.encode_vae(
#                vae=candidate_vae_swap,
#                pixels=inpaint_pixel_fill)['samples']
#
#        progressbar(async_task, 13, 'VAE encoding ...')
#        latent_fill = core.encode_vae(
#            vae=candidate_vae,
#            pixels=inpaint_pixel_fill)['samples']
#
#        inpaint_worker.current_task.load_latent(
#            latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)
#
#        if inpaint_parameterized:
#            pipeline.final_unet = inpaint_worker.current_task.patch(
#                inpaint_head_model_path=inpaint_head_model_path,
#                inpaint_latent=latent_inpaint,
#                inpaint_latent_mask=latent_mask,
#                model=pipeline.final_unet
#            )
#
#        if not advanced_parameters.inpaint_disable_initial_latent:
#            initial_latent = {'samples': latent_fill}
#
#        B, C, H, W = latent_fill.shape
#        height, width = H * 8, W * 8
#        final_height, final_width = inpaint_worker.current_task.image.shape[:2]
#        print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

#    if 'cn' in goals:
#        for task in cn_tasks[flags.cn_canny]:
#            cn_img, cn_stop, cn_weight = task
#            cn_img = resize_image(HWC3(cn_img), width=width, height=height)
#
#            if not advanced_parameters.skipping_cn_preprocessor:
#                cn_img = preprocessors.canny_pyramid(cn_img)
#
#            cn_img = HWC3(cn_img)
#            task[0] = core.numpy_to_pytorch(cn_img)
#            if advanced_parameters.debugging_cn_preprocessor:
#                yield_result(async_task, cn_img, do_not_show_finished_images=True)
#                return
#        for task in cn_tasks[flags.cn_cpds]:
#            cn_img, cn_stop, cn_weight = task
#            cn_img = resize_image(HWC3(cn_img), width=width, height=height)
#
#            if not advanced_parameters.skipping_cn_preprocessor:
#                cn_img = preprocessors.cpds(cn_img)
#
#            cn_img = HWC3(cn_img)
#            task[0] = core.numpy_to_pytorch(cn_img)
#            if advanced_parameters.debugging_cn_preprocessor:
#                yield_result(async_task, cn_img, do_not_show_finished_images=True)
#                return
#        for task in cn_tasks[flags.cn_ip]:
#            cn_img, cn_stop, cn_weight = task
#            cn_img = HWC3(cn_img)
#
#            # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
#            cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
#
#            task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
#            if advanced_parameters.debugging_cn_preprocessor:
#                yield_result(async_task, cn_img, do_not_show_finished_images=True)
#                return
#        for task in cn_tasks[flags.cn_ip_face]:
#            cn_img, cn_stop, cn_weight = task
#            cn_img = HWC3(cn_img)
#
#            if not advanced_parameters.skipping_cn_preprocessor:
#                cn_img = extras.face_crop.crop_image(cn_img)
#
#            # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
#            cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
#
#            task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
#            if advanced_parameters.debugging_cn_preprocessor:
#                yield_result(async_task, cn_img, do_not_show_finished_images=True)
#                return
#
#        all_ip_tasks = cn_tasks[flags.cn_ip] + cn_tasks[flags.cn_ip_face]
#
#        if len(all_ip_tasks) > 0:
#            pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

    if advanced_parameters.freeu_enabled:
        print(f'FreeU is enabled!')
        pipeline.final_unet = core.apply_freeu(
            pipeline.final_unet,
            advanced_parameters.freeu_b1,
            advanced_parameters.freeu_b2,
            advanced_parameters.freeu_s1,
            advanced_parameters.freeu_s2
        )

    print(f'[Parameters] Denoising Strength = {denoising_strength}')

    if isinstance(initial_latent, dict) and 'samples' in initial_latent:
        log_shape = initial_latent['samples'].shape
    else:
        log_shape = f'Image Space {(height, width)}'

    print(f'[Parameters] Initial Latent shape: {log_shape}')

    preparation_time = time.perf_counter() - execution_start_time
    print(f'Preparation time: {preparation_time:.2f} seconds')

    final_sampler_name = sampler_name
    final_scheduler_name = scheduler_name

    if scheduler_name == 'lcm':
        final_scheduler_name = 'sgm_uniform'
        if pipeline.final_unet is not None:
            pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                pipeline.final_unet,
                sampling='lcm',
                zsnr=False)[0]
        if pipeline.final_refiner_unet is not None:
            pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                pipeline.final_refiner_unet,
                sampling='lcm',
                zsnr=False)[0]
        print('Using lcm scheduler.')

    async_task.yields.append(['preview', (13, 'Moving model to GPU ...', None)])

# -------------------------------------------------------------------

    execution_start_time = time.perf_counter()

#    if 'cn' in goals:
#        for cn_flag, cn_path in [
#            (flags.cn_canny, controlnet_canny_path),
#            (flags.cn_cpds, controlnet_cpds_path)
#        ]:
#            for cn_img, cn_stop, cn_weight in cn_tasks[cn_flag]:
#                positive_cond, negative_cond = core.apply_controlnet(
#                    positive_cond, negative_cond,
#                    pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

#    def callback(step, x0, x, total_steps, y):
#        pass


    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps, y):
        pbar.update_absolute(step)

    imgs = pipeline.process_diffusion(
        positive_cond=positive_cond,
        negative_cond=negative_cond,
        steps=steps,
        switch=switch,
        width=width,
        height=height,
        image_seed=seed,
        callback=callback,
        sampler_name=final_sampler_name,
        scheduler_name=final_scheduler_name,
        latent=initial_latent,
        denoise=denoising_strength,
        tiled=tiled,
        cfg_scale=cfg_scale,
        refiner_swap_method=refiner_swap_method
    )

    if inpaint_worker.current_task is not None:
        imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

    execution_time = time.perf_counter() - execution_start_time
    print(f'Generating and saving time: {execution_time:.2f} seconds')

    return core.numpy_to_pytorch(np.array(imgs[0], dtype=np.ubyte))

class FooocusWrapper:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline_in": ("FOOOCUS_PIPELINE", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 2661447138349841858, "min": 0, "max": 0xffffffffffffffff, "step": 1, "display": "number"}),
#                "int_field": ("INT", {
#                    "default": 0,
#                    "min": 0, #Minimum value
#                    "max": 4096, #Maximum value
#                    "step": 64, #Slider's step
#                    "display": "number" # Cosmetic only: display as "number" or "slider"
#                }),
#                "float_field": ("FLOAT", {
#                    "default": 1.0,
#                    "min": 0.0,
#                    "max": 10.0,
#                    "step": 0.01,
#                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
#                    "display": "number"}),
#                "print_to_screen": (["enable", "disable"],),
#                "string_field": ("STRING", {
#                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
#                    "default": "Hello World!"
#                }),
            },
            "optional": {
                "vary_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "Fooocus"

    def process(self, pipeline_in, positive, negative, seed, vary_image = None):
        import modules.advanced_parameters as advanced_parameters

#        adv_args = (False, 1.5, 0.8, 0.3, 7, 'dpmpp_2m_sde_gpu', 'karras', False, -1, -1, -1, -1, -1, -1, False, False, False, False, 0.25, 64, 128, 'joint', False, 1.01, 1.02, 0.99, 0.95, False, False, 'v2.6', 1, 0.618, False, False, 0)
#        advanced_parameters.set_all_advanced_parameters(*adv_args)

        args = ['Speed', '896×1152', 1, 2, 3, 'None', 0.5, 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 0.25, 'None', 1, 'None', 1, 'None', 1, 'None', 1, 'Strong', [], None, '', None, None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt']
        task = AsyncTask(args=list(args))
        image = processTaskSimple(task, pipeline_in, positive, negative, seed, vary_image)

#        args = ['a cat', 'unrealistic, saturated, high contrast, big nose, painting, drawing, sketch, cartoon, anime, manga, render, CG, 3d, watermark, signature, label', ['Fooocus V2', 'Fooocus Photograph', 'Fooocus Negative'], 'Speed', '896×1152 <span style="color: grey;"> ∣ 7:9</span>', 1, '0', 2, 3, 'juggernautXL_v8Rundiffusion.safetensors', 'None', 0.5, 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 0.25, 'None', 1, 'None', 1, 'None', 1, 'None', 1, False, 'uov', 'Disabled', None, [], None, '', None, None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt']
#        task = AsyncTask(args=list(args))
#        image = processTask(task)

        return (image,)

# ----------------------------------------------------------------------------------------------------------

class FooocusPrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline_in": ("FOOOCUS_PIPELINE", ),
                "positive": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "display": "number"})
            },
        }

    RETURN_TYPES = ("FOOOCUS_PIPELINE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipeline_out", "positive", "negative")

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "Fooocus"

    def process(self, pipeline_in, positive, negative, seed):
        import random
        import modules.default_pipeline as pipeline

        from modules.sdxl_styles import apply_style, apply_wildcards, fooocus_expansion
        from extras.expansion import safe_str
        from modules.util import remove_empty_str

        pipeline_out = copy.deepcopy(pipeline_in)

        # TODO
        print("------------------------------------")
        print(f"[Fooocus Prompt]: got {len(pipeline_in['loras'])} loras and {len(pipeline_in['ip_tasks'])} ip-adapter tasks")
        print("------------------------------------")

        prompts = remove_empty_str([safe_str(p) for p in positive.splitlines()], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in negative.splitlines()], default='')

        style_selections = ['Fooocus Photograph', 'Fooocus Negative']

        # TODO: Make an option
        use_expansion = True

        use_style = len(style_selections) > 0

        prompt = prompts[0]
        negative_prompt = negative_prompts[0]

        if prompt == '':
            # disable expansion when empty since it is not meaningful and influences image prompt
            use_expansion = False

        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

#        print('Loading models ...')
#        pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
#                                    loras=loras, base_model_additional_loras=base_model_additional_loras,
#                                    use_synthetic_refiner=use_synthetic_refiner)
#refresh_pipeline(pipeline, pipeline_in)

        print('Processing prompts ...')

        task_seed = seed
        task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future

        task_prompt = apply_wildcards(prompt, task_rng)
        task_negative_prompt = apply_wildcards(negative_prompt, task_rng)
        task_extra_positive_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
        task_extra_negative_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_negative_prompts]

        positive_basic_workloads = []
        negative_basic_workloads = []

        if use_style:
            for s in style_selections:
                p, n = apply_style(s, positive=task_prompt)
                positive_basic_workloads = positive_basic_workloads + p
                negative_basic_workloads = negative_basic_workloads + n
        else:
            positive_basic_workloads.append(task_prompt)

        negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

        positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
        negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

        positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
        negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

        positive_top_k=len(positive_basic_workloads)
        negative_top_k=len(negative_basic_workloads)

        if use_expansion:
            print('Preparing Fooocus text...')
            # TODO: Does this load unnecessary diffusion model?
            expansion = pipeline.final_expansion(task_prompt, task_seed)
            print(f'[Prompt Expansion] {expansion}')
            positive_basic_workloads = copy.deepcopy(positive_basic_workloads) + [expansion]  # Deep copy.

#        print("----------------------------------------------------------------------")
#        print(f"Positive: {positive_basic_workloads}")
#        print("----------------------------------------------------------------------")
#        print(f"Negative: {negative_basic_workloads}")
#        print("----------------------------------------------------------------------")
#        print(f"Positive top_k: {positive_top_k}")
#        print("----------------------------------------------------------------------")
#        print(f"Negative top_k: {negative_top_k}")
#        print("----------------------------------------------------------------------")


        print('Encoding positive ...')
        cond_pos = pipeline.clip_encode(texts=positive_basic_workloads, pool_top_k=positive_top_k)

        print(f'Encoding negative...')
        cond_neg = pipeline.clip_encode(texts=negative_basic_workloads, pool_top_k=negative_top_k)

        return (pipeline_out, cond_pos, cond_neg)

# ----------------------------------------------------------------------------------------------------------

class FooocusPipelineLoader:
    def __init__(self):
#        print("---------------------------------------------")
#        comfy_checkpoints = folder_paths.get_full_path("checkpoints")
#        print("---------------------------------------------")
#        print(comfy_checkpoints)
#        path_checkpoints = config.get_dir_or_set_default('path_checkpoints', comfy_checkpoints)
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
             "ckpt_name": (folder_paths.get_filename_list("fooocus_checkpoints"), )
            },
        }

    RETURN_TYPES = ("FOOOCUS_PIPELINE", )
    RETURN_NAMES = ("pipeline",)

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "Fooocus"

    def process(self, ckpt_name):
        import modules.advanced_parameters as advanced_parameters
        import modules.config as config
        adv_args = (False, 1.5, 0.8, 0.3, 7, 'dpmpp_2m_sde_gpu', 'karras', False, -1, -1, -1, -1, -1, -1, False, False, False, False, 0.25, 64, 128, 'joint', False, 1.01, 1.02, 0.99, 0.95, False, False, 'v2.6', 1, 0.618, False, False, 0)
        advanced_parameters.set_all_advanced_parameters(*adv_args)
        config.default_base_model_name = ckpt_name
        config.default_loras = [['None', 1.0], ['None', 1.0], ['None', 1.0], ['None', 1.0], ['None', 1.0]]
        import modules.default_pipeline as pipeline
#        pipeline.refresh_base_model(ckpt_name)
        p = { "base_model": ckpt_name, "refiner": 'None', "loras": [], "additional_loras": [], "use_synthetic_refiner": False, "ip_tasks": [] }
        # TODO Ari
#p["loras"] = [['SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 0.25], ['None', 1.0], ['None', 1.0], ['None', 1.0], ['None', 1.0]]
#        p["loras"] = [['SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 0.25], ['None', 1.0], ['None', 1.0], ['None', 1.0], ['None', 1.0]]
        refresh_pipeline(pipeline, p)
#        pipeline.refresh_everything(refiner_model_name='None', base_model_name=ckpt_name,
#                loras=[], base_model_additional_loras=[], use_synthetic_refiner=False)

        return (p, )

# ------------------------------------------------------------------------------------------------

class FooocusLoras:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
             "pipeline_in": ("FOOOCUS_PIPELINE", ),
             "lora0": (['None'] + folder_paths.get_filename_list("fooocus_loras"), ),
             "lora0_weight": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.001, "display": "number"}),
             "lora1": (['None'] + folder_paths.get_filename_list("fooocus_loras"), ),
             "lora1_weight": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.001, "display": "number"}),
             "lora2": (['None'] + folder_paths.get_filename_list("fooocus_loras"), ),
             "lora2_weight": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.001, "display": "number"}),
             "lora3": (['None'] + folder_paths.get_filename_list("fooocus_loras"), ),
             "lora3_weight": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.001, "display": "number"}),
            },
        }

    RETURN_TYPES = ("FOOOCUS_PIPELINE", )
    RETURN_NAMES = ("pipeline_out",)

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "Fooocus"

    def process(self, pipeline_in, lora0, lora0_weight, lora1, lora1_weight, lora2, lora2_weight, lora3, lora3_weight):
        import modules.default_pipeline as pipeline

        pipeline_out = copy.deepcopy(pipeline_in)

        # TODO
        print("------------------------------------")
        print(f"[Fooocus Loras]: got {len(pipeline_in['loras'])} loras and {len(pipeline_in['ip_tasks'])} ip-adapter tasks")
        print("------------------------------------")

        pipeline_out["loras"] = [[lora0, lora0_weight], [lora1, lora1_weight], [lora2, lora2_weight], [lora3, lora3_weight]]
        refresh_pipeline(pipeline, pipeline_out)

        return (pipeline_out, )

# ----------------------------------------------------------------------------------------------------------

class FooocusImagePrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline_in": ("FOOOCUS_PIPELINE", ),
                "positive_in": ("CONDITIONING", ),
                "negative_in": ("CONDITIONING", ),
                "image_in": ("IMAGE", ),
                "type": (["ImagePrompt", "PyraCanny", "FaceSwap"], ),
                "stop_at": ("FLOAT", { "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "display": "number"}),
                "weight": ("FLOAT", { "default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.001, "display": "number"}),
                "softness": ("FLOAT", { "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "display": "number"}),
                "threshold_low": ("INT", {"default": 64, "min": 0, "max": 255, "step": 1, "display": "number"}),
                "threshold_high": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("FOOOCUS_PIPELINE", "CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("pipeline_out", "positive_out", "negative_out", "preprocessed")

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "Fooocus"

    def process(self, pipeline_in, positive_in, negative_in, image_in, type, stop_at, weight, softness, threshold_low, threshold_high):
        import numpy as np
        import modules.default_pipeline as pipeline
        import modules.core as core
        import modules.flags as flags
        import modules.config
        import extras.ip_adapter as ip_adapter
        import extras.face_crop
        import extras.preprocessors as preprocessors
        from modules.util import HWC3, resize_image, set_image_shape_ceil
        import modules.advanced_parameters as advanced_parameters

        pipeline_out = copy.deepcopy(pipeline_in)
        positive_out = copy.deepcopy(positive_in)
        negative_out = copy.deepcopy(negative_in)

        # TODO
        print("------------------------------------")
        print(f"[Fooocus Image Prompt]: got {len(pipeline_in['loras'])} loras and {len(pipeline_in['ip_tasks'])} ip-adapter tasks")
        print("------------------------------------")

        advanced_parameters.controlnet_softness = softness
        advanced_parameters.canny_low_threshold = threshold_low
        advanced_parameters.canny_high_threshold = threshold_high

        image = np.asarray(core.pytorch_to_numpy(image_in[0]))

        if type == "FaceSwap":
            clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters('face')
            ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

            image = HWC3(image)
            # TODO: Add setting for face cropping?
#            if not advanced_parameters.skipping_cn_preprocessor:
            image = extras.face_crop.crop_image(image)
            image = resize_image(image, width=224, height=224, resize_mode=0)
            image_out = copy.deepcopy(image)
            image_out = core.numpy_to_pytorch(image_out)
            image = ip_adapter.preprocess(image, ip_adapter_path=ip_adapter_face_path)
            task = [image, stop_at, weight]
            pipeline_out["ip_tasks"].append(task)
        elif type == "ImagePrompt":
            clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
            ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
            image = HWC3(image)
            image = resize_image(image, width=224, height=224, resize_mode=0)
            image = ip_adapter.preprocess(image, ip_adapter_path=ip_adapter_path)
            task = [image, stop_at, weight]
            pipeline_out["ip_tasks"].append(task)
            image_out = image_in
        elif type == "PyraCanny":
            # TODO: How to resize canny correctly?
            shape_ceil = 1024
            image = set_image_shape_ceil(image, shape_ceil)
            image = preprocessors.canny_pyramid(image)
            image = HWC3(image)
            image = core.numpy_to_pytorch(image)
            controlnet_canny_path = modules.config.downloading_controlnet_canny()
            pipeline.refresh_controlnets([controlnet_canny_path])
            positive_out, negative_out = core.apply_controlnet(positive_out, negative_out, pipeline.loaded_ControlNets[controlnet_canny_path], image, weight, 0, stop_at)
            image_out = image


# TODO: Pyracanny thresholds as input
        return (pipeline_out, positive_out, negative_out, image_out)

# ----------------------------------------------------------------------------------------------------------

class FooocusRef:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 2661447138349841858, "min": 0, "max": 0xffffffffffffffff, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "process"

    #OUTPUT_NODE = False

    CATEGORY = "Fooocus"

    def process(self, seed):
        import modules.advanced_parameters as advanced_parameters
        import modules.async_worker as worker
        import modules.core as core
        import numpy as np
        import time

        adv_args = (False, 1.5, 0.8, 0.3, 7, 'dpmpp_2m_sde_gpu', 'karras', False, -1, -1, -1, -1, -1, -1, False, False, False, False, 0.25, 64, 128, 'joint', False, 1.01, 1.02, 0.99, 0.95, False, False, 'v2.6', 1, 0.618, False, False, 0)
        advanced_parameters.set_all_advanced_parameters(*adv_args)

        args = ['a cat', 'unrealistic, saturated, high contrast, big nose, painting, drawing, sketch, cartoon, anime, manga, render, CG, 3d, watermark, signature, label', ['Fooocus V2', 'Fooocus Photograph', 'Fooocus Negative'], 'Speed', '896×1152 <span style="color: grey;"> ∣ 7:9</span>', 1, '0', 2, 3, 'juggernautXL_v8Rundiffusion.safetensors', 'None', 0.5, 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 0.25, 'None', 1, 'None', 1, 'None', 1, 'None', 1, False, 'uov', 'Disabled', None, [], None, '', None, None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt', None, 0.5, 0.6, 'ImagePrompt']

        task = worker.AsyncTask(args=list(args))
        worker.async_tasks.append(task)
        finished = False

        while not finished:
            time.sleep(0.01)
            if len(task.yields) > 0:
                flag, product = task.yields.pop(0)
                if flag == 'preview':
                    pass
                if flag == 'results':
                    print("results")
                if flag == 'finish':
                    print("finish")
                    image = core.numpy_to_pytorch(np.array(product[0], dtype=np.ubyte))
                    finished = True

        return (image,)

# ----------------------------------------------------------------------------------------------------------


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FooocusWrapper": FooocusWrapper,
    "FooocusPrompt": FooocusPrompt,
    "FooocusPipelineLoader": FooocusPipelineLoader,
    "FooocusLoras": FooocusLoras,
    "FooocusImagePrompt": FooocusImagePrompt,
    "FooocusRef": FooocusRef,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FooocusWrapper": "Fooocus Wrapper",
    "FooocusPrompt": "Fooocus Prompt",
    "FooocusPipelineLoader": "Fooocus Pipeline Loader",
    "FooocusLoras": "Fooocus Loras",
    "FooocusImagePrompt": "Fooocus Image Prompt",
    "FooocusRef": "Fooocus Ref",
}
