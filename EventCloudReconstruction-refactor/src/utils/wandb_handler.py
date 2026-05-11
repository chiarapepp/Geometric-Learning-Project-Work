import wandb

class WandbHandler:
    """
    A class to handle logging with Weights & Biases (wandb).
    """
    def __init__(self, config):
        self.config = config

        if self.config.wandb == 'online':
            self.run = wandb.init(project=self.config.wandb_project, entity=self.config.wandb_entity, config=self.config.__dict__)
        elif self.config.wandb == 'disabled':
            self.run = wandb.init(mode='disabled')
        else:
            raise ValueError(f'Unknown wandb mode: {self.args.wandb}')
        
        self.run.log_code(".")  # save the code to wandb
        self.run.define_metric("epoch")
        self.run.define_metric("it")
        self.run.define_metric("train/*", step_metric="epoch")
        self.run.define_metric("val/*", step_metric="epoch")

    def log(self, log_dict):
        self.run.log(log_dict)

    def finish(self):
        self.run.finish()

    def log_video(self, frame_tensor, video_name, caption=''):
        self.run.log({video_name: wandb.Video(frame_tensor, caption=caption)})

    def log_image(self, image,  image_name, caption=''):
        self.run.log({image_name: wandb.Image(image, caption=caption)})

    def log_point_cloud(self, points, cloud_name, it, caption=''):
        self.run.log({cloud_name: wandb.Object3D(points, caption=caption), 'it': it})

    def log_checkpoint(self, model, model_name, epoch):
        self.run.save(model.state_dict(), f"{model_name}_{epoch}.pt")

    