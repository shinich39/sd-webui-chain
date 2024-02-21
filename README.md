## sd-webui-chain

Execute multiple batches sequentially.

## Usage

1. Copy sd-webui-chain to /webui/extensions.
2. Copy the file /webui/sd-webui-chain/loopback_for_chain.py into /webui/scripts directory.
3. Edit webui-user.bat
4. Add "--api" to COMMANDLINE_ARGS (e.g. COMMANDLINE_ARGS=--api)
5. img2img > Change img2img options
6. img2img batch > Open chain
7. Enter batch name > Add
8. Chain tab > Select queue > Generate

## img2img Options

- Checkpoint
- VAE
- Clip Skip
- Prompt
- Negative prompt
- Input directory(Required)
- Output directory(default: /input directory/output)
- Upscale(chain)
- Resize mode
- Sampling method
- Sampling steps
- Refiner
- Batch count
- Batch size
- Width, Height(resize to) or Scale(resize by)
- CFG Scale
- Denoising strength
- Seed
- Override settings(clip skip...)
- (script) ADetailer
- (script) SD upscale
- (script) Loopback for chain

## Environment

- StableDiffusion v1.7.0
- ADetailer v24.1.1
