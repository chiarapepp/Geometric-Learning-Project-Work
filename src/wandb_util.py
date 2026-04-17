class WandbHandler:
    """
    A class to handle logging with Weights & Biases (wandb).
    """
    def __init__(
        self,
        config=None,
        project=None,
        entity=None,
        run_name=None,
        group=None,
        job_type=None,
        tags=None,
    ):
        self.config = config
        self.mode = self._get_config_value("wandb", "disabled")
        self.enabled = self.mode == "online"

        if self.mode == "online":
            import wandb

            self.run = wandb.init(
                project=project or self._get_config_value("wandb_project", "geometric-learning-project"),
                entity=entity if entity is not None else self._get_config_value("wandb_entity", None),
                name=run_name if run_name is not None else self._get_config_value("wandb_run_name", None),
                group=group if group is not None else self._get_config_value("wandb_group", None),
                job_type=job_type if job_type is not None else self._get_config_value("wandb_job_type", None),
                tags=tags if tags is not None else self._get_config_value("wandb_tags", None),
                config=self._config_dict(),
            )
        elif self.mode == 'disabled':
            self.run = None
            return
        else:
            raise ValueError(f'Unknown wandb mode: {self.mode}')
        
        self.run.log_code(".")  # save the code to wandb
        self.run.define_metric("epoch")
        self.run.define_metric("it")
        self.run.define_metric("train/*", step_metric="epoch")
        self.run.define_metric("val/*", step_metric="epoch")
        self.run.define_metric("benchmark/*")

    def _get_config_value(self, name, default=None):
        if self.config is None:
            return default
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return getattr(self.config, name, default)

    def _config_dict(self):
        if self.config is None:
            return {}
        if isinstance(self.config, dict):
            return dict(self.config)
        if hasattr(self.config, "__dict__"):
            return dict(self.config.__dict__)
        return {}

    def log(self, log_dict):
        if self.run is None:
            return
        self.run.log(log_dict)

    def finish(self):
        if self.run is None:
            return
        self.run.finish()

    def log_video(self, frame_tensor, video_name, caption=''):
        if self.run is None:
            return
        import wandb
        self.run.log({video_name: wandb.Video(frame_tensor, caption=caption)})

    def log_image(self, image,  image_name, caption=''):
        if self.run is None:
            return
        import wandb
        self.run.log({image_name: wandb.Image(image, caption=caption)})

    def log_point_cloud(self, points, cloud_name, it, caption=''):
        if self.run is None:
            return
        import wandb
        self.run.log({cloud_name: wandb.Object3D(points, caption=caption), 'it': it})

    def log_table(self, table_name, rows):
        if self.run is None:
            return
        if not rows:
            return
        import wandb

        columns = list(rows[0].keys())
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row.get(column) for column in columns])
        self.run.log({table_name: table})

    def log_artifact(self, path, name=None, artifact_type="result"):
        if self.run is None:
            return
        import os
        import wandb

        artifact_name = name or os.path.splitext(os.path.basename(path))[0]
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def log_benchmark_results(self, rows, csv_path=None, table_name="benchmark/results"):
        if self.run is None:
            return
        if not rows:
            return

        by_loss = {}
        for row in rows:
            by_loss.setdefault(row["loss"], []).append(row)

        metrics = {}
        for loss_name, loss_rows in by_loss.items():
            values = [float(row["value"]) for row in loss_rows]
            seconds = [float(row["seconds"]) for row in loss_rows]
            memory = [float(row["peak_memory_mb"]) for row in loss_rows]
            prefix = f"benchmark/{loss_name}"
            metrics[f"{prefix}/mean_value"] = sum(values) / len(values)
            metrics[f"{prefix}/mean_seconds"] = sum(seconds) / len(seconds)
            metrics[f"{prefix}/peak_memory_mb"] = max(memory)
        self.log(metrics)
        self.log_table(table_name, rows)
        if csv_path is not None:
            self.log_artifact(csv_path, artifact_type="benchmark-csv")

    def log_checkpoint(self, checkpoint_path, artifact_name=None):
        if self.run is None:
            return
        self.log_artifact(checkpoint_path, name=artifact_name, artifact_type="model")

    
