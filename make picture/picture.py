from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# pipe.to("cuda")  # این خط رو حذف کن!
image = pipe("a girl with curlly and brown hair with wings in heaven").images[0]
image.save("result.png")
