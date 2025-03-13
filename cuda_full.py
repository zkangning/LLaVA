# cuda_occupancy_daemon.py
import torch
import time
import sys

class MemoryOccupier:
    def __init__(self, target_ratio=0.9):
        self.target_ratio = target_ratio
        self.memory_blocks = []
        self.device = self._validate_device()

    def _validate_device(self):
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA-capable device detected")
        return torch.device("cuda")

    def _calculate_allocation(self):
        total = torch.cuda.get_device_properties(self.device).total_memory
        allocated = torch.cuda.memory_allocated(self.device)
        return max(0, int(total * self.target_ratio) - allocated)

    def occupy_memory(self):
        chunk_size = 256 * 1024 * 1024  # 256MB chunks
        target_mem = self._calculate_allocation()
        
        print(f"[Init] Device: {torch.cuda.get_device_name(self.device)}")
        print(f"[Init] Target occupation: {self.target_ratio*100}%")

        while target_mem > 0:
            allocate_size = min(chunk_size, target_mem)
            try:
                self.memory_blocks.append(
                    torch.empty(allocate_size, dtype=torch.uint8, device=self.device)
                )
                target_mem -= allocate_size
                print(f"Allocated {allocate_size//1024**2}MB "
                      f"({len(self.memory_blocks)} chunks)")
            except RuntimeError as e:
                print(f"Allocation halted: {str(e)}")
                break

    def maintain_occupation(self):
        print("Entering maintenance mode...")
        while True:
            time.sleep(300)  # 5分钟心跳检测
            current_ratio = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(self.device).total_memory
            if current_ratio < self.target_ratio * 0.95:
                print("Detected memory decrease, reallocating...")
                self.occupy_memory()

if __name__ == "__main__":
    try:
        occupier = MemoryOccupier(target_ratio=0.9)
        occupier.occupy_memory()
        occupier.maintain_occupation()
    except KeyboardInterrupt:
        print("\nDaemon terminated by user")
    except Exception as e:
        print(f"Critical error: {str(e)}")
    finally:
        torch.cuda.empty_cache()
        print("Memory resources released")

# setenv CUDA_VISIBLE_DEVICES 1 nohup python3 cuda_occupancy_daemon.py > /dev/null 2>&1 &