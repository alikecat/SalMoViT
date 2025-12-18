import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from salmovit_model import SalMoViT
from config import ModelConfig
from rich.console import Console
from rich.table import Table


def analyze_model_complexity(
    model, modules=None, centerbias=False, static=False, sizes=[224, 256, 320, 384, 512]
):
    results = {"params": {}, "flops": {}}
    model.eval()
    param_count = parameter_count(model)
    total_params = param_count[""]
    if modules:
        for name, module in modules.items():
            if isinstance(module, list):
                params = sum(sum(p.numel() for p in m.parameters()) for m in module)
            else:
                params = sum(p.numel() for p in module.parameters())
            results["params"][name] = params
    results["params"]["total"] = total_params
    for size in sizes:
        x = torch.randn(1, 3, size, size)
        flops = (
            FlopCountAnalysis(model, (x, torch.randn(1, size, size)))
            if centerbias
            else (
                FlopCountAnalysis(model, (x.unsqueeze(0)))
                if static
                else FlopCountAnalysis(model, x)
            )
        )
        results["flops"][f"{size}x{size}"] = flops.total()
    return results


def print_results(results, model_name="Model"):
    console = Console()
    console.print(f"\n[bold cyan]{'=' * 50}[/bold cyan]")
    console.print(f"[bold cyan]{model_name:^50}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 50}[/bold cyan]")
    table = Table(title="Parameters", show_header=True, header_style="bold magenta")
    table.add_column("Module", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    for name, params in results["params"].items():
        if name != "total":
            percentage = (params / results["params"]["total"]) * 100
            table.add_row(name, f"{params:,}", f"{percentage:.1f}%")

    table.add_row("Total", f"{results['params']['total']:,}", "100.0%")
    console.print(table)
    model_size = results["params"]["total"] * 4 / (1024 * 1024)
    console.print(f"\n[bold]Model size:[/bold] {model_size:.2f} MB")
    flops_table = Table(title="FLOPs", show_header=True, header_style="bold magenta")
    flops_table.add_column("Input Size", style="cyan")
    flops_table.add_column("FLOPs (G)", justify="right", style="green")
    for size, flops in sorted(
        results["flops"].items(), key=lambda x: int(x[0].split("x")[0])
    ):
        flops_table.add_row(size, f"{flops / 1e9:.2f}")
    console.print(flops_table)


if __name__ == "__main__":
    model = SalMoViT(preprocess=False)
    modules = {
        "encoder": model.encoder,
        "bottleneck": model.bottleneck,
        "upsample": [model.up0, model.up1, model.up2, model.up3, model.up4],
        "output": model.out_conv,
    }
    salmovit_results = analyze_model_complexity(model, modules)
    # model = ModelConfig.DEEPGAZE_IIE.get_model()
    # deepgaze_iie_results = analyze_model_complexity(model, centerbias=True)
    # model = ModelConfig.UNISAL.get_model()
    # unisal_results = analyze_model_complexity(model, static=True)
    print_results(salmovit_results, "SalMoViT")
    # print_results(deepgaze_iie_results, "DeepGaze IIE")
    # print_results(unisal_results, "UniSal")
