import os
import torch
from aligngen.aligned_pipeline import FluxPipeline
from aligngen.attention_processor import ShareAttnFluxAttnProcessor2_0, StyleAlignedArgs
from aligngen.aligned_transformer import FluxTransformer2DModel
from PIL import Image
import torch
import numpy as np
import argparse


def init_attention_processors(pipeline: FluxPipeline, style_aligned_args: StyleAlignedArgs | None = None):
    attn_procs = {}
    transformer = pipeline.transformer
    for i, name in enumerate(transformer.attn_processors.keys()):
        attn_procs[name] = ShareAttnFluxAttnProcessor2_0(
            cnt=i,
            style_aligned_args=style_aligned_args,
        )
    transformer.set_attn_processor(attn_procs)


def concat_img(images, k):
    img_width, img_height = images[0].size
    n = len(images)
    rows = (n + k - 1) // k

    total_width = img_width * k
    total_height = img_height * rows
    new_img = Image.new('RGB', (total_width, total_height), 'white')

    for i, img in enumerate(images):
        row = i // k
        col = i % k
        new_img.paste(img, (col * img_width, row * img_height))

    return new_img


def main(args):
    # --- 1. Setup AlignedGen Arguments ---
    style_args = StyleAlignedArgs(
        share_attention=True,
        block=(19, 57),
        timesteps=(0, 30),
        style_lambda_mode="fix",
        style_lambda=args.style_lambda,  # Use the parsed argument here
        constrain_first=True
    )

    # --- 2. Load Model ---
    print(f"Loading model from: {args.model_path}")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    transformer.set_style_aligned_args(style_args)

    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    init_attention_processors(pipe, style_args)
    pipe = pipe.to('cuda')

    # --- 3. Run Inference ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating images...")
    images = pipe(
        prompts,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images
    torch.cuda.empty_cache()

    # --- 4. Save Results ---
    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f"{i}.jpg"))

    concat_result = concat_img(images, len(images))
    concat_result.save(f"{output_dir}/concat.jpg")
    print(f"Results saved to '{output_dir}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlignedGen Inference")

    parser.add_argument(
        "--model_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path or Hugging Face ID of the pretrained FLUX model."
    )
    parser.add_argument(
        "--style_lambda",
        type=float,
        default=1.1,
        help="The lambda value for controlling style alignment strength."
    )
    prompts = [
        "Anchor in 3D realism style.",
        "Clock in 3D realism style.",
        "Globe in 3D realism style.",
        "Bicycle in 3D realism style.",
    ]
    parsed_args = parser.parse_args()
    main(parsed_args)
