import os
import json
import requests
import base64

from PIL import Image
from json import JSONDecodeError
from operator import itemgetter

import gradio as gr

from modules import shared, script_callbacks, scripts
from modules.ui_components import InputAccordion, FormRow, ToolButton
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.ui import switch_values_symbol
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae

BASE_DIR = scripts.basedir() # /extensions/sd-webui-queue
CACHE_DIR_NAME = "caches"
QUEUES_FILE_NAME = "queues.json"
BATCHES_FILE_NAME = "batches.json"
COMPONENTS_FILE_NAME = "components.json"

def create_cache_dir():
  # path = Path(paths.module_path, CACHE_DIR_NAME)
  if not os.path.exists(f"{BASE_DIR}/{CACHE_DIR_NAME}"):
    os.makedirs(f"{BASE_DIR}/{CACHE_DIR_NAME}")
    
def write_json(json_data, file_name: str):
  create_cache_dir()
  with open(f"{BASE_DIR}/{CACHE_DIR_NAME}/{file_name}", "w") as file:
    file.write(json.dumps(json_data, indent=2))

def write_text(new_text, file_name: str):
  create_cache_dir()
  with open(f"{BASE_DIR}/{CACHE_DIR_NAME}/{file_name}", "w") as file:
    file.write(new_text)

def replace_text(old: str, new: str, file_name: str):
  with open(f"{BASE_DIR}/{CACHE_DIR_NAME}/{file_name}", "w") as file:
    json_data = file.read()

  with open(f"{BASE_DIR}/{CACHE_DIR_NAME}/{file_name}", "w") as file:
    file.write(json_data.replace(old, new))

def load_json(file_name, default_data = []):
  try:
    with open(f"{BASE_DIR}/{CACHE_DIR_NAME}/{file_name}") as file:
      json_data = json.load(file)

  except (FileNotFoundError, JSONDecodeError) as err:
    json_data = default_data

    write_json(json_data, file_name)

    if err.__class__ == FileNotFoundError:
      print(f"{file_name} not found.")
    elif err.__class__ == JSONDecodeError:
      print(f"{file_name} was corrupt or malformed.")

  return json_data

def save_json(json_data, file_name):

  write_json(json_data, file_name)

  print(f"{file_name} has been saved.")

def load_component_list():
  with open(f"{BASE_DIR}/{COMPONENTS_FILE_NAME}") as file:
    json_data = json.load(file)

  for key in list(json_data.keys()):
    if not json_data[key]:
      # print(f"Delete {key}: {json_data[key]}")
      del(json_data[key])

  # print(f"load_component_list {json_data}")
  return json_data

def click_add_batch_button(batch_id, *args):
  global batches

  # check dupe
  for batch in batches:
    if batch["id"] == batch_id:
      raise Exception(f"{batch_id} already exists.")

  req = {}
  for index, value in enumerate(args):
    key = component_keys[index]
    req[key] = value
    # print(f"Init: {key}: {value}")

  payload = {}
  # payload["init_images"] = [] # base64
  payload["prompt"] = req["prompt"]
  payload["negative_prompt"] = req["negative_prompt"]
  # payload["styles"] = []
  payload["seed"] = req["seed"]

  if "subseed_show" in req and req["subseed_show"]:
    payload["subseed"] = req["subseed"]
    payload["subseed_strength"] = req["subseed_strength"]
    payload["seed_resize_from_w"] = req["seed_resize_from_w"]
    payload["seed_resize_from_h"] = req["seed_resize_from_h"]

  payload["resize_tab"] = req["resize_tab"] # 0, 1 / by, to
  payload["width"] = req["resize_width"] # to
  payload["height"] = req["resize_height"] # to
  payload["resize_scale"] = req["resize_scale"] # by

  payload["resize_mode"] = req["resize_mode"]
  payload["cfg_scale"] = req["cfg_scale"]
  payload["denoising_strength"] = req["denoising_strength"]
  payload["steps"] = req["steps"]
  payload["sampler_name"] = req["sampler_name"]
  payload["n_iter"] = req["n_iter"]
  payload["batch_size"] = req["batch_size"]
  payload["clip_skip"] = shared.opts.CLIP_stop_at_last_layers
  # payload["tiling"] = True
  # payload["restore_faces"] = False
  payload["do_not_save_samples"] = True
  payload["do_not_save_grid"] = True
  # payload["eta"] = 0
  # payload["s_min_uncond"] = 0
  # payload["s_churn"] = 0
  # payload["s_tmax"] = 0
  # payload["s_tmin"] = 0
  # payload["s_noise"] = 0

  if "refiner_enable" in req and req["refiner_enable"]:
    payload["refiner_checkpoint"] = req["refiner_checkpoint"]
    payload["refiner_switch_at"] = req["refiner_switch_at"]

  if "script_list" in req and req["script_list"] and not req["script_list"] == 0:
    try:
      script_list = [script.name for script in scripts.scripts_img2img.scripts if script.name is not None]
      script_name = script_list[req["script_list"] - 1]

      print(f"Script found: {script_name}")

      if script_name.lower() == "sd upscale":
        payload["script_name"] = script_name
        # self, p, _, overlap, upscaler_index, scale_factor
        payload["script_args"] = [
          None,
          req["sd_upscale_overlap"],
          req["sd_upscale_upscaler_index"],
          req["sd_upscale_scale_factor"],
        ]

      elif script_name.lower() == "loopback":
        payload["script_name"] = script_name
        # self, p, loops, final_denoising_strength, denoising_curve, append_interrogation
        payload["script_args"] = [
          req["loopback_loops"],
          req["loopback_final_denoising_strength"],
          # req["denoising_curve"],
          "Linear",
          # req["append_interrogation"],
          "None",
        ]

      elif script_name.lower() == "loopback for chain":
        payload["script_name"] = script_name
        # self, p, loops, final_denoising_strength, denoising_curve, append_interrogation
        payload["script_args"] = [
          req["loopback_for_chain_loops"],
          req["loopback_for_chain_final_denoising_strength"],
          req["loopback_for_chain_denoising_curve"],
          req["loopback_for_chain_append_interrogation"],
        ]

      # elif script_name.lower() == "x/y/z plot":
      #   payload["script_name"] = script_name
      #   payload["script_args"] = [
      #     req["xyz_plot_x_type"],
      #     req["xyz_plot_x_values"],
      #     req["xyz_plot_y_type"],
      #     req["xyz_plot_y_values"],
      #     req["xyz_plot_z_type"],
      #     req["xyz_plot_z_values"],
      #     req["xyz_plot_draw_legend"],
      #     req["xyz_plot_include_lone_images"],
      #     req["xyz_plot_include_sub_grids"],
      #     req["xyz_plot_no_fixed_seeds"],
      #     req["xyz_plot_margin_size"],
      #     req["xyz_plot_csv_mode"],
      #   ]
    except Exception as err:
      print(f"Script not found.")


  # payload["disable_extra_networks"] = False
  # payload["comments"] = {}
  # payload["send_images"] = True
  # payload["save_images"] = False
  
  sd_model_name = shared.sd_model.sd_checkpoint_info.name_for_extra
  sd_model_info = modules.sd_models.get_closet_checkpoint_match(sd_model_name)
  if sd_model_info is None:
    raise RuntimeError(f"Unknown checkpoint: {sd_model_name}")
  
  payload["override_settings"] = {}
  payload["override_settings_restore_afterwards"] = True
  payload["override_settings"]['sd_model_checkpoint'] = sd_model_info.name
  payload["override_settings"]["sd_vae"] = modules.sd_vae.get_loaded_vae_name() or "None"
  # payload["override_settings"]["CLIP_stop_at_last_layers"] = shared.opts.CLIP_stop_at_last_layers
  # payload["override_settings"]["sd_model_hash"] = shared.sd_model.sd_model_hash
  # payload["override_settings"]["sd_model_name"] = shared.sd_model.sd_checkpoint_info.name_for_extra
  # payload["override_settings"]["sd_vae_hash"] = modules.sd_vae.get_loaded_vae_hash()
  # payload["override_settings"]["sd_vae_name"] = modules.sd_vae.get_loaded_vae_name()

  # override settings
  if "img2img_override_settings" in req and req["img2img_override_settings"]:
    override_settings = create_override_settings_dict(req["img2img_override_settings"])
    # merge override settings
    payload["override_settings"] = dict(payload["override_settings"], **override_settings)

  # scripts
  payload["alwayson_scripts"] = {}
  
  # adetailer
  if "ad_enable" in req and req["ad_enable"]:
    payload["alwayson_scripts"]["ADetailer"] = {
      "args": [
        True,
        False,
        {
          "ad_model": req["ad_model"],
          "ad_prompt": req["ad_prompt"],
          "ad_negative_prompt": req["ad_negative_prompt"],
          "ad_confidence": req["ad_confidence"],
          "ad_mask_min_ratio": req["ad_mask_min_ratio"],
          "ad_mask_max_ratio": req["ad_mask_max_ratio"],
          "ad_x_offset": req["ad_x_offset"],
          "ad_y_offset": req["ad_y_offset"],
          "ad_dilate_erode": req["ad_dilate_erode"],
          "ad_mask_merge_invert": req["ad_mask_merge_invert"],
          "ad_mask_blur": req["ad_mask_blur"],
          "ad_denoising_strength": req["ad_denoising_strength"],
          "ad_inpaint_only_masked": req["ad_inpaint_only_masked"],
          "ad_inpaint_only_masked_padding": req["ad_inpaint_only_masked_padding"],
          "ad_use_inpaint_width_height": req["ad_use_inpaint_width_height"],
          "ad_inpaint_width": req["ad_inpaint_width"],
          "ad_inpaint_height": req["ad_inpaint_height"],
          "ad_use_steps": req["ad_use_steps"],
          "ad_steps": req["ad_steps"],
          "ad_use_cfg_scale": req["ad_use_cfg_scale"],
          "ad_cfg_scale": req["ad_cfg_scale"],
          "ad_restore_face": req["ad_restore_face"],
          "ad_controlnet_model": req["ad_controlnet_model"],
          "ad_controlnet_weight": req["ad_controlnet_weight"],
          "ad_controlnet_guidance_start": req["ad_controlnet_guidance_start"],
          "ad_controlnet_guidance_end": req["ad_controlnet_guidance_end"],
        }, 
        {
          "ad_model": req["ad_model_2nd"],
          "ad_prompt": req["ad_prompt_2nd"],
          "ad_negative_prompt": req["ad_negative_prompt_2nd"],
          "ad_confidence": req["ad_confidence_2nd"],
          "ad_mask_min_ratio": req["ad_mask_min_ratio_2nd"],
          "ad_mask_max_ratio": req["ad_mask_max_ratio_2nd"],
          "ad_x_offset": req["ad_x_offset_2nd"],
          "ad_y_offset": req["ad_y_offset_2nd"],
          "ad_dilate_erode": req["ad_dilate_erode_2nd"],
          "ad_mask_merge_invert": req["ad_mask_merge_invert_2nd"],
          "ad_mask_blur": req["ad_mask_blur_2nd"],
          "ad_denoising_strength": req["ad_denoising_strength_2nd"],
          "ad_inpaint_only_masked": req["ad_inpaint_only_masked_2nd"],
          "ad_inpaint_only_masked_padding": req["ad_inpaint_only_masked_padding_2nd"],
          "ad_use_inpaint_width_height": req["ad_use_inpaint_width_height_2nd"],
          "ad_inpaint_width": req["ad_inpaint_width_2nd"],
          "ad_inpaint_height": req["ad_inpaint_height_2nd"],
          "ad_use_steps": req["ad_use_steps_2nd"],
          "ad_steps": req["ad_steps_2nd"],
          "ad_use_cfg_scale": req["ad_use_cfg_scale_2nd"],
          "ad_cfg_scale": req["ad_cfg_scale_2nd"],
          "ad_restore_face": req["ad_restore_face_2nd"],
          "ad_controlnet_model": req["ad_controlnet_model_2nd"],
          "ad_controlnet_weight": req["ad_controlnet_weight_2nd"],
          "ad_controlnet_guidance_start": req["ad_controlnet_guidance_start_2nd"],
          "ad_controlnet_guidance_end": req["ad_controlnet_guidance_end_2nd"],
        },
      ]
    }

  # post-process upscale
  upscale_payload = None
  if "upscale_enable" in req and req["upscale_enable"]:
    upscale_payload = {
      "resize_mode": 0,
      "show_extras_results": False,
      "gfpgan_visibility": 0,
      "codeformer_visibility": 0,
      "codeformer_weight": 0,
      "upscaling_crop": req["upscale_crop"],
      "upscaler_1": req["upscale_upscaler_1"],
      "upscaler_2": req["upscale_upscaler_2"],
      "extras_upscaler_2_visibility": req["upscale_upscaler_2_visibility"],
      "upscale_first": False, # upscale before restoring faces
      # "imageList": [] # [{ "data": "base64", "name": "filename"}]
    }

    if req["upscale_tab"] == 0:
      # 0: by
      upscale_payload["upscaling_resize"] = req["upscale_scale"]
    else:
      # 1: to
      upscale_payload["upscaling_resize_w"] = req["upscale_width"]
      upscale_payload["upscaling_resize_h"] = req["upscale_height"]

  input_dir = req["batch_input_dir"]
  if input_dir is None:
    raise RuntimeError(f"input_dir not found.")
  
  output_dir = req["batch_output_dir"]
  if not output_dir:
    output_dir = os.path.join(input_dir, "output")

  batches.append({
    "id": batch_id,
    "input_dir": input_dir,
    "output_dir": output_dir,
    "payload": payload,
    "upscale_payload": upscale_payload,
  })

  batches = sorted(batches, key=itemgetter("id"))

  save_json(batches, BATCHES_FILE_NAME)

  print(f"Create a new batch: {batch_id}")

  return [
    gr.Textbox.update(
      value=""
    ),
    gr.Dropdown.update(
      choices=[o['id'] for o in batches],
    ),
    gr.Dropdown.update(
      choices=[o['id'] for o in batches],
    )
  ]

def click_remove_batch_btn(batch_id, selected_batch_ids):
  global batches, queues

  for index, batch in enumerate(batches):
    if batch["id"] == batch_id:
      batches.pop(index)
      break
  else:
    raise Exception(f"{batch_id} not found.")
  
  for index, id in enumerate(selected_batch_ids):
    if id == batch_id:
      selected_batch_ids.pop(index)
      break

  for queue in queues:
    for index, id in enumerate(queue["batches"]):
      if id == batch_id:
        queue["batches"].pop(index)
        break

  save_json(queues, QUEUES_FILE_NAME)
  save_json(batches, BATCHES_FILE_NAME)

  print(f"Remove a batch: {batch_id}")

  return [
    gr.Dropdown.update(
      choices=[o['id'] for o in batches],
      value=None
    ),
    gr.Dropdown.update(
      choices=[o['id'] for o in batches],
      value=selected_batch_ids
    )
  ]

def click_save_queue_btn(queue_id, batch_list):
  global queues

  # check dupe
  for queue in queues:
    if queue["id"] == queue_id:
      raise Exception(f"{queue_id} already exists.")

  queues.append({
    "id": queue_id,
    "batches": batch_list
  })

  queues = sorted(queues, key=itemgetter("id"))

  save_json(queues, QUEUES_FILE_NAME)

  print(f"Create a new queue: {queue_id}")

  return [
    gr.Textbox.update(
      value="",
    ),
    gr.Dropdown.update(),
    gr.Dropdown.update(
      choices=[o['id'] for o in queues],
      value=queue_id
    ),
  ]

def click_remove_queue_btn(queue_id):
  global queues

  for index, queue in enumerate(queues):
    if queue["id"] == queue_id:
      break
  else:
    raise Exception(f"{queue_id} not found.")
    
  queues.pop(index)

  save_json(queues, QUEUES_FILE_NAME)

  print(f"Remove a queue: {queue_id}")

  return gr.Dropdown.update(
    choices=[o['id'] for o in queues],
    value="None"
  )

def change_queue_ipt(batch_ids):
  batch_id = batch_ids.pop()
  batch = next((x for x in batches if x["id"] == batch_id), None)

  if not batch:
    raise Exception(f"{batch_id} not found.")

  # print(f"Change queue: {batch}")

  return gr.TextArea.update(
    label=f"Batch: {batch_id}",
    value=json.dumps(batch, indent=2),
  )

def change_queues_ipt(queue_id):
  queue = next((x for x in queues if x["id"] == queue_id), None)

  if not queue:
    raise Exception(f"{queue_id} not found.")

  # print(f"Change queue list: {queue}")

  return gr.Dropdown.update(
    value=queue["batches"],
  )

def change_batches_ipt(batch_id):
  batch = next((x for x in batches if x["id"] == batch_id), None)

  if not batch:
    raise Exception(f"{batch_id} not found.")

  # print(f"Change batch list: {batch}")

  return gr.TextArea.update(
    label=f"Batch: {batch_id}",
    value=json.dumps(batch, indent=2),
  )

def click_generate_btn_start():
  return [
    gr.Button.update(
      visible=False
    ),
    gr.Button.update(
      visible=True
    ),
  ]

def click_generate_btn_end():
  return [
    gr.Button.update(
      visible=True
    ),
    gr.Button.update(
      visible=False
    ),
  ]

def click_generate_btn(queue):
  global in_progress

  in_progress = True
  result = ""
  exts = [".jpg",".jpeg",".gif",".png"]

  print(f"Queue: {queue}")

  for batch_id in queue:
    if not in_progress:
      print(f"Process interrupted.")
      result += f"Process interrupted.\n"
      return [
        gr.Button.update(
          visible=True
        ),
        gr.Button.update(
          visible=False
        ),
        gr.TextArea.update(
          label="Result",
          value=result
        )
      ]

    batch = next((x for x in batches if x["id"] == batch_id), None)
    if not batch:
      print(f"Batch {batch_id} not found.")
      result += f"Batch {batch_id} not found.\n"
      continue
    
    print(f"Batch: {batch_id}")

    input_dir = batch["input_dir"]
    output_dir = batch["output_dir"]
    img2img_payload = batch["payload"]
    upscale_payload = batch["upscale_payload"]
    
    for filename in os.listdir(input_dir):
      basename = os.path.splitext(filename)[0]
      extension = os.path.splitext(filename)[1]
      payload = img2img_payload.copy()
      payload["init_images"] = []

      if not extension.lower() in exts:
        continue
      
      input_path = os.path.join(input_dir, filename)

      if payload["resize_tab"] == 1:
        intput_width, input_height = Image.open(input_path).size
        image_width = intput_width * payload["resize_scale"]
        image_height = input_height * payload["resize_scale"]
        payload["width"] = image_width
        payload["height"] = image_height

      with open(input_path, 'rb') as input_file:
        input_data = input_file.read()
        input_base64 = base64.b64encode(input_data).decode('utf-8')
        payload["init_images"].append(input_base64) # push 

      print(f"Processing: {input_path}")
      # result += f"Processing: {input_path}\n"
      
      response = requests.post("http://127.0.0.1:7860/sdapi/v1/img2img", json=payload)
      res = response.json()

      if not response.status_code == 200:
        print(f"Generate-Error: {input_path}")
        result += f"Generate-Error: {input_path}\n"
        print(res)
        continue

      print(f"Generated: {input_path}")
      # result += f"Generated: {input_path}\n"

      # post-process upscale
      if not upscale_payload == None:
        upscale_payload["imageList"] = []
        
        for index, image in enumerate(res['images']):
          upscale_payload["imageList"].append({
            "data": image,
            "name": f"{basename}_{str(index).zfill(4)}.png"
          })

        upscale_response = requests.post("http://127.0.0.1:7860/sdapi/v1/extra-batch-images", json=upscale_payload)
        upscale_res = upscale_response.json()

        if not upscale_response.status_code == 200:
          print(f"Upscale-Error: {input_path}")
          result += f"Upscale-Error: {input_path}\n"
          print(upscale_res)
          continue

        print(f"Upscaled: {input_path}")
        # result += f"Upscaled: {input_path}\n"

        processed_images = upscale_res['images']
      else:
        processed_images = res['images']
        
      # check dir
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)

      # save
      for index, image in enumerate(processed_images):
        if not in_progress:
          print(f"Process interrupted.")
          result += f"Process interrupted.\n"
          return [
            gr.Button.update(
              visible=True
            ),
            gr.Button.update(
              visible=False
            ),
            gr.TextArea.update(
              label="Result",
              value=result
            )
          ]
        
        new_filename = f"{basename}_{str(index).zfill(4)}.png"
        output_path = os.path.join(output_dir, new_filename)
        try:
          output_data = base64.b64decode(image)
          with open(output_path, 'wb') as output_file:
            output_file.write(output_data)
            
          print(f"Saved: {output_path}")
          result += f"Saved: {output_path}\n"
        except Exception as err:
          print(f"Write-Error: {output_path}")
          result += f"Write-error: {output_path}\n"
          print(err)

  in_progress = False

  return [
    gr.Button.update(
      visible=True
    ),
    gr.Button.update(
      visible=False
    ),
    gr.TextArea.update(
      label="Result",
      value=result
    )
  ]

def click_interrupt_btn(*args):
  global in_progress
  in_progress = False
  print(f"Chain interrupt.")
  return

component_list = load_component_list()
component_map = []
component_keys = []
batches = load_json(BATCHES_FILE_NAME, [])
queues = load_json(QUEUES_FILE_NAME, [])
add_batch_ipt = None
add_batch_btn = None
result_ipt = None
in_progress = False

def on_after_component(component, **kwargs):
  global add_batch_ipt, add_batch_btn

  if component.elem_id == "img2img_batch_inpaint_mask_dir":
    with gr.Accordion("Chain", open=False):
      selected_scale_tab = gr.State(value=0)

      gr.Markdown("""
        <div>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, nisl eget ultricies aliquam, nunc nisl aliquet nunc, eget aliquam nisl nunc vel nisl.
        </div>
      """)

      with InputAccordion(False, label="Upscale") as upscale_enable:
        with gr.Column():
          with FormRow():
            with gr.Tabs():
              # 0
              with gr.TabItem("Scale by") as upscale_scale_by:
                upscale_scale = gr.Slider(
                  minimum=1,
                  maximum=8,
                  step=1,
                  label="Resize",
                  value=4,
                )
              # 1
              with gr.TabItem("Scale to") as upscale_scale_to:
                with FormRow():
                  with gr.Column(scale=4):
                    upscale_width = gr.Slider(
                      minimum=64,
                      maximum=2048,
                      step=8,
                      label="Width",
                      value=512,
                    )
                    upscale_height = gr.Slider(
                      minimum=64,
                      maximum=2048,
                      step=8,
                      label="Height",
                      value=512,
                    )
                  with gr.Column(scale=1, elem_classes="dimensions-tools"):
                    with FormRow():
                      upscale_switch = ToolButton(value=switch_values_symbol, tooltip="Switch width/height")
                    with FormRow():
                      upscale_crop = gr.Checkbox(label='Crop to fit', value=True)

          with FormRow():
            upscaler_1 = gr.Dropdown(
              label='Upscaler 1',
              choices=[x.name for x in shared.sd_upscalers],
              value=shared.sd_upscalers[0].name,
            )

          with FormRow():
            upscaler_2 = gr.Dropdown(
              label='Upscaler 2',
              choices=[x.name for x in shared.sd_upscalers],
              value=shared.sd_upscalers[0].name,
            )

            upscaler_2_visibility = gr.Slider(
              minimum=0.0,
              maximum=1.0,
              step=0.001,
              label="Upscaler 2 visibility",
              value=0.0,
            )

      with FormRow():
        add_batch_ipt = gr.Textbox(label="Batch name")
        add_batch_btn = gr.Button("Add", scale=0, min_width=63)
        
      upscale_switch.click(lambda w, h: (h, w), inputs=[upscale_width, upscale_height], outputs=[upscale_width, upscale_height], show_progress=False)
      upscale_scale_by.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
      upscale_scale_to.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])

      component_map.append(upscale_enable)
      component_keys.append("upscale_enable")
      component_map.append(selected_scale_tab)
      component_keys.append("upscale_tab")
      component_map.append(upscale_scale)
      component_keys.append("upscale_scale")
      component_map.append(upscale_width)
      component_keys.append("upscale_width")
      component_map.append(upscale_height)
      component_keys.append("upscale_height")
      component_map.append(upscale_crop)
      component_keys.append("upscale_crop")
      component_map.append(upscaler_1)
      component_keys.append("upscale_upscaler_1")
      component_map.append(upscaler_2)
      component_keys.append("upscale_upscaler_2")
      component_map.append(upscaler_2_visibility)
      component_keys.append("upscale_upscaler_2_visibility")

  if not component.elem_id == None and component.elem_id in component_list:
    if component.elem_id == "img2img_tab_resize_by":
      # 0: resize_to / 1: resize_by
      component_map.append(component.parent.parent.children[0]) # resize_tab
      component_keys.append("resize_tab")
    # elif component.elem_id == "script_loopback_final_denoising_strength":
    #   print(component.parent.parent.children)
    #   # Denoising strength curve
    #   component_map.append(component.parent.children[2].children[1].children[1].children[0].children[0])
    #   component_keys.append("denoising_curve")
    #   # Append interrogated prompt at each iteration
    #   component_map.append(component.parent.children[3].children[1].children[1].children[0].children[0]) 
    #   component_keys.append("append_interrogation")
    else:
      component_map.append(component)
      component_keys.append(component_list[component.elem_id])

def on_ui_tab(**kwargs):
  global result_ipt

  with gr.Blocks(analytics_enabled=False) as container:
    with gr.Row():
      queue_ipt = gr.Dropdown(
        choices=[o['id'] for o in batches],
        multiselect=True,
        interactive=True,
        allow_custom_value=False,
        scale=1,
        label="Queue",
        info="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, nisl eget ultricies aliquam, nunc nisl aliquet nunc, eget aliquam nisl nunc vel nisl.",
      )

      generate_btn = gr.Button("Generate", variant="primary", scale=0, min_width=320, visible=True)
      interrupt_btn = gr.Button("Interrupt", variant="stop", scale=0, min_width=320, visible=False)
    with gr.Row():
      with gr.Column():
        with gr.Row():
          save_queue_ipt = gr.Textbox(
            label="Save a queue",
            scale=1,
          )

          save_queue_btn = gr.Button("Save", scale=0, min_width=80)

        with gr.Row():
          queues_ipt = gr.Dropdown(
            choices=[o['id'] for o in queues],
            value="None",
            scale=1,
            label="Queue List",
          )
          
          remove_queue_btn = gr.Button("Remove", variant="stop", scale=0, min_width=80)

        with gr.Row():
          batches_ipt = gr.Dropdown(
            choices=[o['id'] for o in batches],
            value=None,
            scale=1,
            label="Batch List",
          )
        
          remove_batch_btn = gr.Button("Remove", variant="stop", scale=0, min_width=80)

      with gr.Column():
        result_ipt = gr.TextArea("", label="Result")

    queue_ipt.select(change_queue_ipt, queue_ipt, result_ipt)
    queues_ipt.select(change_queues_ipt, queues_ipt, queue_ipt)
    batches_ipt.select(change_batches_ipt, batches_ipt, result_ipt)
    save_queue_btn.click(click_save_queue_btn, [save_queue_ipt, queue_ipt], [save_queue_ipt, queue_ipt, queues_ipt])
    remove_queue_btn.click(click_remove_queue_btn, queues_ipt, queues_ipt)
    add_batch_btn.click(click_add_batch_button, [add_batch_ipt, *component_map], [add_batch_ipt, batches_ipt, queue_ipt])
    remove_batch_btn.click(click_remove_batch_btn, [batches_ipt, queue_ipt], [batches_ipt, queue_ipt])
    generate_btn.click(
      click_generate_btn_start, 
      None, 
      [generate_btn, interrupt_btn]
    ).then(
      click_generate_btn, 
      queue_ipt,
      [generate_btn, interrupt_btn, result_ipt]
    )

    interrupt_btn.click(click_interrupt_btn, None, None)

  return [(container, "Chain", "chain")]

class Script(scripts.Script):

  def __init__(self):
    super().__init__()

  def title(self):
    return "Chain"

  def show(self, is_img2img):
    return False # return scripts.AlwaysVisible
    
  def ui(self, is_img2img):
    pass

script_callbacks.on_after_component(on_after_component)
script_callbacks.on_ui_tabs(on_ui_tab)