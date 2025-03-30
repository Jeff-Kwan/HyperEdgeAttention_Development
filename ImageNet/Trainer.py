'''
DataDistributedParallel Trainer to be used with torchrun.
'''
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from tqdm import tqdm
from time import time
import traceback


def distributed_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
        sampler=DistributedSampler(dataset, shuffle=True, drop_last=False))


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader=None,
    ) -> None:
        
        # Initialize the Trainer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = self._get_loss()

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.model = model.to(self.rank)
        self.model = torch.compile(self.model)    # Titan too old for compilation
        self.model = DDP(self.model, device_ids=[self.rank])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=len(train_loader), gamma=config.gamma) if config.gamma else None

        # Useful variables
        self.output_dir = config.output_dir
        self.running_loss = 0.0
        self.epoch_loss = 0.0
        self.at_step = 0

        # Snapshot management
        os.makedirs(self.output_dir, exist_ok=True)
        self.snapshot_path = os.path.join(self.output_dir, "snapshot.pt")
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)


    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.rank}"
        snapshot = torch.load(snapshot_path, map_location=loc, weights_only=True)
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        if self.scheduler:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.config.at_epoch = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.config.at_epoch}")

        # Synchronize all processes to ensure consistency
        torch.distributed.barrier()

    def _save_snapshot(self):
        # Model snapshot
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()  # Correctly save the underlying model
        snapshot["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        snapshot["SCHEDULER_STATE"] = self.scheduler.state_dict() if self.scheduler else None
        snapshot["EPOCHS_RUN"] = self.config.at_epoch
        torch.save(snapshot, self.snapshot_path)

        # Auxiliary information
        self.config.train_losses.append(self.running_loss / self.at_step)
        self.running_loss = 0.0
        self.at_step = 0
        # self.config.val_losses.append(self.validate())
        self.config.train_time += time() - self.start_time
        self.start_time = time()
        save_model_desc(self.output_dir, self.config, self.model)
        plot_losses(self.config, self.output_dir)
        if self.rank == 0:
            val = f"Val Loss: {self.config.val_losses[-1]:.5f}" if len(self.config.val_losses) > 0 else ""
            tqdm.write(f"Epoch [{self.config.at_epoch}/{self.config.epochs}], Loss: {self.config.train_losses[-1]:.5f}," + val)


    def _get_loss(self):
        loss = self.config.loss.strip().lower()
        if loss in ['bcewithlogitsloss', 'bce']:
            criterion = torch.nn.BCEWithLogitsLoss()
        elif loss in ['crossentropyloss', 'crossentropy', 'ce']:
            criterion = torch.nn.CrossEntropyLoss()
        elif loss in ['mseloss', 'mse']:
            criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f"Loss function {loss} not implemented.")
        return criterion


    def _model_forward(self, batch_x, batch_y):
        if self.config.autocast:
            if self.config.autocast == 'fp16':
                dtype = torch.float16
            elif self.config.autocast == 'bf16':
                dtype = torch.bfloat16
            elif self.config.autocast == 'tf32':
                dtype = torch.float32
            else:
                raise ValueError("Invalid mixed precision type.")
            
            with torch.autocast('cuda', dtype=dtype):
                logits = self.model(batch_x)
                if hasattr(self.config, "vocab_size"):
                    loss = self.criterion(logits.view(-1, self.config.vocab_size), batch_y.view(-1))
                else:
                    loss = self.criterion(logits, batch_y)
            loss.backward()
        else:
            logits = self.model(batch_x)
            if hasattr(self.config, "vocab_size"):
                loss = self.criterion(logits.view(-1, self.config.vocab_size), batch_y.view(-1))
            else:
                loss = self.criterion(logits, batch_y)
            loss.backward()
        if torch.isnan(loss).any():
            raise ValueError("Loss is NaN.")
        return loss

    def _run_batch(self, batch_x, batch_y, p_bar):
        batch_x = batch_x.to(self.rank)
        batch_y = batch_y.to(self.rank)
        self.optimizer.zero_grad(set_to_none=True)

        # Minibatching
        if self.config.minibatch:
            for mb in range((len(batch_x)+self.config.minibatch-1)//self.config.minibatch):
                mini_x = batch_x[mb*self.config.minibatch:(mb+1)*self.config.minibatch]
                mini_y = batch_y[mb*self.config.minibatch:(mb+1)*self.config.minibatch]
                loss = self._model_forward(mini_x, mini_y)
        else:
            loss = self._model_forward(batch_x, batch_y)

        # Clip norm and backpropagate
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.running_loss += loss.item()
        self.epoch_loss += loss.item()
        if self.config.gamma:
            self.scheduler.step()

        p_bar.set_postfix({'Norm': norm.item(), 'Loss': loss.item()})
        if self.config.save_steps and (self.config.b_iter < self.config.tot_batches):
            if self.rank == 0 and self.at_step == self.config.save_steps:
                self._save_snapshot()


    def _run_epoch(self, epoch):
        # print(f"[GPU{self.rank}] Epoch {epoch} | Batchsize: {self.config.batch} | Steps: {self.config.b_iter}")
        self.train_loader.sampler.set_epoch(epoch)
        self.config.b_iter = 0
        self.config.at_epoch += 1
        self.model.train()
        p_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for batch_x, batch_y in p_bar:
            self._run_batch(batch_x, batch_y, p_bar)
            self.at_step += 1
            self.config.b_iter += 1
        if self.rank == 0:
            self._save_snapshot()


    def crunch(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.rank)
        batch_x, batch_y = next(iter(self.train_loader))
        batch_x = batch_x.to(self.rank)
        batch_y = batch_y.to(self.rank)
        self.start_time = time()
        p_bar = tqdm(self.train_loader)
        for i in range(self.config.epochs):
            self._run_batch(batch_x, batch_y, p_bar)
        print(f"Time taken: {time()-self.start_time:.5f} s")
        if torch.cuda.is_available():
            print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(self.rank) / 1e6:.2f} MB")


    def train(self, max_retries: int = 5):
        """
        Start the training process. If an exception occurs, load the last snapshot and retry.
        
        Args:
            max_retries (int): Maximum number of retry attempts. Default is 5.
        """
        assert self.config.minibatch is None or self.config.batch > self.config.minibatch or self.config.batch % self.config.minibatch == 0, "Invalid batch-minibatch configuration."
        self.config.autocast = self.config.autocast.strip().lower() if self.config.autocast else None        

        self.config.tot_batches = len(self.train_loader)
        self.start_time = time()
        retries = 0

        while self.config.at_epoch < self.config.epochs and retries < max_retries:
            # Initialize the exception flag tensor to 0
            exception_flag = torch.tensor(0, device=self.rank)
            try:
                current_epoch = self.config.at_epoch
                self._run_epoch(current_epoch)
            except Exception as e:
                # Log the exception details
                error_message = f"Exception occurred during training at epoch {self.config.at_epoch}: {str(e)}"
                print(error_message)
                traceback.print_exc()

                # Set the exception flag to 1
                exception_flag = torch.tensor(1, device=self.rank)
            
            # Aggregate the exception flags across all processes
            torch.distributed.all_reduce(exception_flag, op=torch.distributed.ReduceOp.MAX)
            
            if exception_flag.item() > 0:
                # An exception has occurred in at least one process
                if self.rank == 0:
                    print("An exception has occurred in one of the processes.")

                # Increment retry counter
                retries += 1
                if retries > max_retries:
                    if self.rank == 0:
                        print(f"Maximum retry limit reached ({max_retries}). Exiting training.")
                    break
                else:
                    if self.rank == 0:
                        print(f"Attempting to recover... (Retry {retries}/{max_retries})")

                # Load the last saved snapshot to resume training
                if os.path.exists(self.snapshot_path):
                    self._load_snapshot(self.snapshot_path)
                    if self.rank == 0:
                        print(f"Resuming training from epoch {self.config.at_epoch}.")
                    exception_raised = True  # Prevent raising the exception again
                else:
                    if self.rank == 0:
                        print("No snapshot found. Cannot resume training.")
                    break

            # Synchronize all processes to ensure consistency
            try:
                torch.distributed.barrier()
            except Exception as e:
                print(f"Exception during barrier synchronization: {e}")
                break

        if retries >= max_retries:
            if self.rank == 0:
                print(f"Maximum retry limit reached ({max_retries}). Exiting training.")
        else:
            if self.rank == 0:
                print("Training completed successfully.")
