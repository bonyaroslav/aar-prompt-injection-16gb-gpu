# WSL RAM and GPU-memory limits

## Bottom line

The `50%` WSL value is a **default maximum for WSL 2 system RAM**, not a permanent reservation and not a GPU-VRAM formula. It can be raised with `%UserProfile%\.wslconfig`.

The apparently related `~7.5 GB` GPU figure is usually the Windows **shared system-memory capacity** reported for the GPU. It is separate from WSL and is not RAM that WSL has reserved for VRAM. Windows uses a GPU shared-memory policy of roughly half of host RAM, dynamically lending and reclaiming pages. It does **not** mean “50% of WSL RAM” or “25% of host RAM.”

## What the two numbers mean

| Number | Owner | Default / behaviour | Can WSL `memory=` change it? |
| --- | --- | --- | --- |
| WSL 2 RAM | WSL utility VM | Default maximum: **50% of all Windows memory**. WSL memory grows/shrinks with use. | **Yes** |
| GPU shared system memory | Windows WDDM / GPU driver | A dynamic host-DRAM budget, commonly around **50% of host RAM**. It is an upper budget, not pre-reserved RAM. | **No** |
| Dedicated GPU VRAM | Physical GPU + Windows driver | Real VRAM is shared/budgeted by Windows across Windows and WSL clients. The available budget can change while other GPU work runs. | **No** |

For example, a `7.5 GB` *shared GPU memory* value is consistent with about `15 GB` usable host system memory under the Windows graphics-memory formula. It is not evidence that WSL first received 15 GB and then gave half of that to the GPU.

## Evidence

- Microsoft documents `[wsl2] memory` as “How much memory to assign to the WSL 2 VM,” with a default of **50% of total Windows memory**. It documents `.wslconfig` as global to WSL 2 distributions and supplies no GPU-memory/VRAM setting. [WSL advanced settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- WSL GPU access uses GPU paravirtualization: multiple VMs share the hardware; the guest has no video-memory manager; requests are marshalled to the host. This is not a fixed GPU-memory slice derived from the WSL VM RAM limit. [Microsoft GPU-PV architecture](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-paravirtualization)
- Microsoft's DirectX team states the WSL consequence directly: Linux applications have the same GPU access as native Windows applications; there is no Windows/Linux resource partition or Linux-specific limit, and the sharing is dynamic. A lone Linux application can consume all available GPU resources. [DirectX ❤ Linux](https://devblogs.microsoft.com/directx/directx-heart-linux/)
- Windows documents the shared-graphics-memory calculation. It caps graphics system memory at a host-memory-derived amount (often half of host RAM), then exposes the remainder as maximum shared GPU memory. [Calculating graphics memory](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/calculating-graphics-memory)
- Microsoft’s Task Manager explanation is explicit: shared GPU memory is normal system memory used flexibly by the CPU or GPU; Windows dynamically locks/releases pages and allows the GPU to use up to half of physical RAM at a time. [GPUs in Task Manager](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)
- The GPU's actual local/non-local memory budget is provided by the OS and can change with activity from other processes. [DXGI video-memory budget](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info)
- NVIDIA documents different WSL CUDA constraints (notably limited full Unified Memory and limited pinned system memory); it does not document an adjustable WSL VRAM percentage. [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## Practical configuration for RAM-heavy WSL work

Use a manual WSL 2 ceiling above the 50% default when Linux is the primary workload and Windows is genuinely idle:

```ini
# %UserProfile%\.wslconfig
[wsl2]
memory=24GB
swap=8GB

[experimental]
autoMemoryReclaim=dropCache
```

Then apply it with `wsl --shutdown` and start WSL again. `memory` is a **maximum**, not an upfront allocation; it applies to all WSL 2 distributions for that Windows user. Choose a ceiling that leaves Windows headroom for the desktop, browser, drivers, and GPU data staging; do not set it merely to all installed RAM. NVIDIA’s WSL guidance specifically shows raising this manual ceiling (for example, `memory=32GB`) when the default 50% prevents the workload from using enough system RAM. [NVIDIA NIM on WSL troubleshooting](https://docs.nvidia.com/nim/wsl2/1.0.0/troubleshooting.html)

Increasing `memory=` can resolve a Linux/WSL RAM or swap bottleneck. It **will not increase dedicated VRAM**, nor does it raise Windows' shared-GPU-memory policy. If the error is CUDA out-of-memory, first check actual VRAM use and the GPU process/budget; if it is Linux OOM or heavy swap, raise the WSL `memory` ceiling.

## WSL lifecycle: boot, launch, and idle shutdown

**WSL does not normally run a distro just because Windows booted.** The WSL 2 utility VM is managed automatically and is started when something launches or uses it. `wsl -d Ubuntu` is explicitly the command to **run** that distribution, so it starts Ubuntu (and, if necessary, the WSL 2 VM). Starting Ubuntu from Start, opening a WSL terminal/VS Code Remote-WSL session, running `wsl <command>`, or a Windows scheduled task that invokes `wsl.exe` have the same practical effect. [WSL commands](https://learn.microsoft.com/en-us/windows/wsl/basic-commands) and [WSL architecture FAQ](https://github.com/MicrosoftDocs/WSL/blob/main/WSL/faq.yml)

| Situation | Result |
| --- | --- |
| Windows boot, with no program/task configured to invoke WSL | WSL stays stopped. |
| `wsl -d Ubuntu` / `wsl <command>` / Ubuntu launcher | Starts that distribution and the shared WSL 2 VM if it is not already running. |
| `wsl --shutdown` | Immediately terminates **all** running WSL distributions and the WSL 2 utility VM. It is the reliable way to stop it before changing `.wslconfig`. |
| Idle WSL 2 VM | The documented default `vmIdleTimeout` is **60,000 ms (60 seconds)** before WSL shuts the idle VM down. |

The timeout is not an instruction to reserve memory for a minute; it is a shutdown grace period. Microsoft also documents that the VM can shut down automatically when it no longer has open Windows-process file handles. Therefore, do not use an ordinary systemd service as a keep-alive mechanism: Microsoft explicitly says **systemd services do not keep a WSL instance alive**. A `[boot] command=` or a systemd unit runs *when an instance is launched*; neither makes WSL auto-start at Windows boot. [WSL advanced settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-config), [systemd in WSL](https://learn.microsoft.com/en-us/windows/wsl/systemd), and [WSL FAQ](https://github.com/MicrosoftDocs/WSL/blob/main/WSL/faq.yml)

**Common apparent "auto-start" causes:**

- A Task Scheduler task, startup item, IDE, or script invokes `wsl.exe`; that is a genuine launch trigger.
- Docker Desktop's WSL backend runs in its own `docker-desktop` WSL distribution. If Docker Desktop is configured to start at Windows sign-in, it will start that backend; this does not by itself mean that Ubuntu was launched. Docker Desktop's own “start at sign-in” option is disabled by default. [Docker Desktop WSL backend](https://docs.docker.com/desktop/features/wsl/) and [Docker Desktop settings](https://docs.docker.com/desktop/settings-and-maintenance/settings/)

## Long-running sessions

WSL 2 normally returns memory freed by processes to Windows, but Microsoft notes that cached pages can persist during a long-running session. Current WSL configuration also offers `autoMemoryReclaim` (default `dropCache`) to reclaim cached memory. [WSL version comparison](https://learn.microsoft.com/en-us/windows/wsl/compare-versions) and [WSL advanced settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
