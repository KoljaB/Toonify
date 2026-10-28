"""`toonify` command line: run offline stages individually or all at once, and serve the player."""
from __future__ import annotations

import argparse
import socket
from pathlib import Path

from .config import RUNS_DIR, Run, run_id_for


def _host_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def player_url(run: Run, port: int) -> str:
    return f"http://{_host_ip()}:{port}/?run={run.id}"


def _run_for(args) -> Run:
    if getattr(args, "run", None):
        return Run(RUNS_DIR / args.run)
    photo = Path(args.photo)
    return Run(RUNS_DIR / run_id_for(photo, args.style))


def main(argv=None):
    p = argparse.ArgumentParser(prog="toonify")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, photo=True):
        if photo:
            sp.add_argument("photo")
            sp.add_argument("--style", default="default")
        sp.add_argument("--force", action="store_true", help="ignore the stage cache")

    sp = sub.add_parser("run", help="stylize + decompose + expressions, then serve the player")
    common(sp)
    sp.add_argument("--backend", default="comfy-qwen", choices=["comfy-qwen", "api"])
    sp.add_argument("--vram", default="auto", choices=["auto", "full", "group_offload", "nf4", "blockswap"])
    sp.add_argument("--port", type=int, default=8780)
    sp.add_argument("--no-serve", action="store_true", help="only print the player URL")

    sp = sub.add_parser("stylize", help="stage 1: photo -> transparent anime PNG")
    common(sp)
    sp.add_argument("--backend", default="comfy-qwen", choices=["comfy-qwen", "api"])
    sp.add_argument("--seed", type=int, default=7)

    sp = sub.add_parser("decompose", help="stage 2: see-through -> avatar.psd + layers.json")
    sp.add_argument("run", help="run id (folder name under runs/)")
    sp.add_argument("--vram", default="auto", choices=["auto", "full", "group_offload", "nf4", "blockswap"])
    sp.add_argument("--force", action="store_true")

    sp = sub.add_parser("expressions", help="stage 3: eyes closed / mouth half / mouth open frames")
    sp.add_argument("run")
    sp.add_argument("--only", nargs="*", help="regenerate only these frames")
    sp.add_argument("--force", action="store_true")

    sp = sub.add_parser("serve", help="serve the player and the audio WebSocket relay")
    sp.add_argument("--port", type=int, default=8780)
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--ssl-cert", help="TLS certificate (AudioWorklet needs https unless opened via localhost)")
    sp.add_argument("--ssl-key")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        from .server import serve
        serve(args.host, args.port, args.ssl_cert, args.ssl_key)
        return

    from .comfy import ComfyClient
    from .stages import decompose, expressions, stylize

    if args.cmd == "stylize":
        run = _run_for(args)
        stylize.run(run, Path(args.photo), args.style, args.backend, args.seed, args.force)
        print(f"run id: {run.id}")
    elif args.cmd == "decompose":
        decompose.run(_run_for(args), args.vram, args.force, before_gpu=ComfyClient().free)
    elif args.cmd == "expressions":
        expressions.run(_run_for(args), args.force, only=args.only)
    elif args.cmd == "run":
        run = _run_for(args)
        client = ComfyClient()
        stylize.run(run, Path(args.photo), args.style, args.backend, force=args.force, client=client)
        decompose.run(run, args.vram, args.force, before_gpu=client.free)
        expressions.run(run, args.force, client=client)
        client.free()
        client.stop()
        url = player_url(run, args.port)
        print(f"\nrun id:     {run.id}\nplayer URL: {url}")
        if not args.no_serve:
            from .server import serve
            serve("0.0.0.0", args.port)


if __name__ == "__main__":
    main()
