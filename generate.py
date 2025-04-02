from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import os
import cv2
import numpy as np
from PIL import Image

class generateToImg:
    def __init__(self, adapter_name, charac_name, prompt, negative_prompt, fix_prompt=None, charac=False):
        self.adapter_name = adapter_name
        self.charac_name = charac_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.fix_prompt = fix_prompt
        self.charac = charac
        self.output_dir = os.path.normpath("tmp/img")

    def generate(self):
        try:
            # 创建基础输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 构建完整输出路径
            output_path = os.path.join(self.output_dir, f"scene_{self.adapter_name}")
            if self.charac:
                charac_dir = os.path.join(self.output_dir, f"charac_{self.charac_name}")
                os.makedirs(charac_dir, exist_ok=True)
                output_path = os.path.join(charac_dir, f"scene_{self.adapter_name}")
            
            os.makedirs(output_path, exist_ok=True)

            # 加载ControlNet模型(当需要角色形象时)
            controlnet = None
            if self.charac:
                controlnet = ControlNetModel.from_single_file(
                    os.path.normpath("models/stablediffusion/diffusion_pytorch_model.safetensors"),
                    torch_dtype=torch.float16
                ).to(self.device)

            # 加载主模型
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                os.path.normpath("models/stablediffusion/meinamix_meinaV11.safetensors"),
                torch_dtype=torch.float16,
                controlnet=controlnet,
                safety_checker=None
            ).to(self.device)

            # 生成控制图像(当需要角色形象时)
            control_image = None
            if self.charac:
                # 先生成初始角色图像
                generator = torch.Generator(self.device).manual_seed(42)
                init_image = pipe(
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt,
                    generator=generator,
                    num_inference_steps=20
                ).images[0]
                
                # 保存初始角色图像
                init_image.save(os.path.join(output_path, "charac_init.png"))
                
                # 转换为灰度控制图像
                np_image = np.array(init_image)
                gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
                gray_image = cv2.Canny(gray_image, 100, 200)
                gray_image = gray_image[:, :, None]
                gray_image = np.concatenate([gray_image, gray_image, gray_image], axis=2)
                control_image = Image.fromarray(gray_image)
                control_image.save(os.path.join(output_path, "charac_control.png"))

            # 生成最终图像
            generator = torch.Generator(self.device).manual_seed(42)
            images = pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                generator=generator,
                num_inference_steps=30,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.5 if self.charac else None,
                image=control_image if self.charac else None
            ).images

            # 保存结果
            for i, img in enumerate(images):
                img.save(os.path.join(output_path, f"output_{i}.png"))
            
            return True

        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return False
