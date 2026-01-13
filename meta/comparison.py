"""
Module de comparaison des résultats entre différentes métaheuristiques.
Fournit des tableaux formatés et des graphiques comparatifs.
"""

from __future__ import annotations

import os
import csv
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_results_csv(csv_path: str) -> List[Dict]:
    """Charge les résultats depuis le fichier CSV."""
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["rcapt"] = int(row["rcapt"])
            row["rcom"] = int(row["rcom"])
            row["time_s"] = float(row["time_s"])
            row["sensors"] = int(row["sensors"])
            row["feasible"] = row["feasible"].lower() == "true"
            row["uncovered"] = int(row["uncovered"])
            row["disconnected"] = int(row["disconnected"])
            results.append(row)
    return results


def organize_by_instance(results: List[Dict]) -> Dict[str, Dict[Tuple[int, int], Dict[str, Dict]]]:
    """
    Organise les résultats par instance, puis par paire (Rcapt, Rcom), puis par algorithme.
    
    Returns:
        {instance_name: {(rcapt, rcom): {algo: {metrics...}}}}
    """
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for row in results:
        instance = row["file"]
        pair = (row["rcapt"], row["rcom"])
        algo = row["algo"]
        
        organized[instance][pair][algo] = {
            "sensors": row["sensors"],
            "time_s": row["time_s"],
            "feasible": row["feasible"],
            "uncovered": row["uncovered"],
            "disconnected": row["disconnected"],
        }
    
    return dict(organized)


def print_summary_table(results: List[Dict], use_pandas: bool = False):
    """Affiche un tableau récapitulatif comparant GRASP et VNS par instance."""
    organized = organize_by_instance(results)
    
    if use_pandas and HAS_PANDAS:
        _print_summary_pandas(organized)
    else:
        _print_summary_tabulate(organized)


def _print_summary_tabulate(organized: Dict):
    """Affiche le tableau avec tabulate."""
    if not HAS_TABULATE:
        print("tabulate non disponible, utilisation du format simple")
        _print_summary_simple(organized)
        return
    
    table_data = []
    headers = ["Instance", "(Rcapt, Rcom)", "GRASP", "VNS", "Meilleur"]
    
    for instance in sorted(organized.keys()):
        for pair in sorted(organized[instance].keys()):
            row = [
                instance,
                f"({pair[0]},{pair[1]})",
            ]
            
            grasp_data = organized[instance][pair].get("grasp", {})
            vns_data = organized[instance][pair].get("vns", {})
            
            grasp_str = "-"
            if grasp_data:
                grasp_str = f"{grasp_data['sensors']} capteurs ({grasp_data['time_s']:.2f}s)"
            
            vns_str = "-"
            if vns_data:
                vns_str = f"{vns_data['sensors']} capteurs ({vns_data['time_s']:.2f}s)"
            
            row.append(grasp_str)
            row.append(vns_str)
            
            # Déterminer le meilleur
            best_str = "-"
            if grasp_data and vns_data:
                if grasp_data["sensors"] < vns_data["sensors"]:
                    best_str = "GRASP"
                elif vns_data["sensors"] < grasp_data["sensors"]:
                    best_str = "VNS"
                else:
                    best_str = "Égalité"
            elif grasp_data:
                best_str = "GRASP"
            elif vns_data:
                best_str = "VNS"
            
            row.append(best_str)
            table_data.append(row)
    
    print("\n" + "="*100)
    print("TABLEAU RÉCAPITULATIF DES RÉSULTATS")
    print("="*100)
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))
    print()


def _print_summary_pandas(organized: Dict):
    """Affiche le tableau avec pandas."""
    rows = []
    for instance in sorted(organized.keys()):
        for pair in sorted(organized[instance].keys()):
            grasp_data = organized[instance][pair].get("grasp", {})
            vns_data = organized[instance][pair].get("vns", {})
            
            # Formater les valeurs GRASP
            grasp_sensors = grasp_data.get("sensors") if grasp_data else None
            grasp_time = f"{grasp_data.get('time_s', 0):.2f}s" if grasp_data else "-"
            grasp_str = f"{grasp_sensors} capteurs ({grasp_time})" if grasp_sensors is not None else "-"
            
            # Formater les valeurs VNS
            vns_sensors = vns_data.get("sensors") if vns_data else None
            vns_time = f"{vns_data.get('time_s', 0):.2f}s" if vns_data else "-"
            vns_str = f"{vns_sensors} capteurs ({vns_time})" if vns_sensors is not None else "-"
            
            row = {
                "Instance": instance,
                "(Rcapt, Rcom)": f"({pair[0]},{pair[1]})",
                "GRASP": grasp_str,
                "VNS": vns_str,
            }
            
            # Meilleur
            if grasp_data and vns_data:
                if grasp_data["sensors"] < vns_data["sensors"]:
                    row["Meilleur"] = "GRASP"
                elif vns_data["sensors"] < grasp_data["sensors"]:
                    row["Meilleur"] = "VNS"
                else:
                    row["Meilleur"] = "Égalité"
            elif grasp_data:
                row["Meilleur"] = "GRASP"
            elif vns_data:
                row["Meilleur"] = "VNS"
            else:
                row["Meilleur"] = "-"
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    print("\n" + "="*120)
    print("TABLEAU RÉCAPITULATIF DES RÉSULTATS")
    print("="*120)
    print(df.to_string(index=False))
    print()


def _print_summary_simple(organized: Dict):
    """Affiche un tableau simple sans dépendances."""
    print("\n" + "="*100)
    print("TABLEAU RÉCAPITULATIF DES RÉSULTATS")
    print("="*100)
    print(f"{'Instance':<40} {'(Rcapt,Rcom)':<15} {'GRASP':<30} {'VNS':<30} {'Meilleur':<10}")
    print("-"*100)
    
    for instance in sorted(organized.keys()):
        for pair in sorted(organized[instance].keys()):
            grasp_data = organized[instance][pair].get("grasp", {})
            vns_data = organized[instance][pair].get("vns", {})
            
            grasp_str = "-"
            if grasp_data:
                grasp_str = f"{grasp_data['sensors']} capteurs ({grasp_data['time_s']:.2f}s)"
            
            vns_str = "-"
            if vns_data:
                vns_str = f"{vns_data['sensors']} capteurs ({vns_data['time_s']:.2f}s)"
            
            best_str = "-"
            if grasp_data and vns_data:
                if grasp_data["sensors"] < vns_data["sensors"]:
                    best_str = "GRASP"
                elif vns_data["sensors"] < grasp_data["sensors"]:
                    best_str = "VNS"
                else:
                    best_str = "Égalité"
            elif grasp_data:
                best_str = "GRASP"
            elif vns_data:
                best_str = "VNS"
            
            print(f"{instance:<40} {f'({pair[0]},{pair[1]})':<15} {grasp_str:<30} {vns_str:<30} {best_str:<10}")
    print()


def print_statistics(results: List[Dict]):
    """Affiche des statistiques comparatives entre les algorithmes."""
    grasp_results = [r for r in results if r["algo"] == "grasp"]
    vns_results = [r for r in results if r["algo"] == "vns"]
    
    print("\n" + "="*80)
    print("STATISTIQUES COMPARATIVES")
    print("="*80)
    
    if grasp_results:
        grasp_sensors = [r["sensors"] for r in grasp_results]
        grasp_times = [r["time_s"] for r in grasp_results]
        
        print("\nGRASP:")
        print(f"  Nombre de capteurs:")
        print(f"    Moyenne: {statistics.mean(grasp_sensors):.2f}")
        print(f"    Médiane: {statistics.median(grasp_sensors):.2f}")
        print(f"    Min: {min(grasp_sensors)}")
        print(f"    Max: {max(grasp_sensors)}")
        if len(grasp_sensors) > 1:
            print(f"    Écart-type: {statistics.stdev(grasp_sensors):.2f}")
        print(f"  Temps de résolution:")
        print(f"    Moyenne: {statistics.mean(grasp_times):.2f}s")
        print(f"    Médiane: {statistics.median(grasp_times):.2f}s")
        print(f"    Min: {min(grasp_times):.2f}s")
        print(f"    Max: {max(grasp_times):.2f}s")
    
    if vns_results:
        vns_sensors = [r["sensors"] for r in vns_results]
        vns_times = [r["time_s"] for r in vns_results]
        
        print("\nVNS:")
        print(f"  Nombre de capteurs:")
        print(f"    Moyenne: {statistics.mean(vns_sensors):.2f}")
        print(f"    Médiane: {statistics.median(vns_sensors):.2f}")
        print(f"    Min: {min(vns_sensors)}")
        print(f"    Max: {max(vns_sensors)}")
        if len(vns_sensors) > 1:
            print(f"    Écart-type: {statistics.stdev(vns_sensors):.2f}")
        print(f"  Temps de résolution:")
        print(f"    Moyenne: {statistics.mean(vns_times):.2f}s")
        print(f"    Médiane: {statistics.median(vns_times):.2f}s")
        print(f"    Min: {min(vns_times):.2f}s")
        print(f"    Max: {max(vns_times):.2f}s")
    
    # Comparaison directe si on a les deux
    if grasp_results and vns_results:
        organized = organize_by_instance(results)
        grasp_wins = 0
        vns_wins = 0
        ties = 0
        
        for instance in organized:
            for pair in organized[instance]:
                grasp_data = organized[instance][pair].get("grasp", {})
                vns_data = organized[instance][pair].get("vns", {})
                
                if grasp_data and vns_data:
                    if grasp_data["sensors"] < vns_data["sensors"]:
                        grasp_wins += 1
                    elif vns_data["sensors"] < grasp_data["sensors"]:
                        vns_wins += 1
                    else:
                        ties += 1
        
        print(f"\nComparaison directe:")
        print(f"  GRASP meilleur: {grasp_wins} fois")
        print(f"  VNS meilleur: {vns_wins} fois")
        print(f"  Égalité: {ties} fois")
    
    print()


def plot_comparison_bar_chart(results: List[Dict], save_path: Optional[str] = None, show: bool = True):
    """Génère un graphique en barres comparant GRASP et VNS par instance."""
    organized = organize_by_instance(results)
    
    instances = []
    pairs = []
    grasp_values = []
    vns_values = []
    
    for instance in sorted(organized.keys()):
        for pair in sorted(organized[instance].keys()):
            instances.append(f"{instance}\nR=({pair[0]},{pair[1]})")
            pairs.append(pair)
            
            grasp_data = organized[instance][pair].get("grasp", {})
            vns_data = organized[instance][pair].get("vns", {})
            
            grasp_values.append(grasp_data.get("sensors", 0) if grasp_data else 0)
            vns_values.append(vns_data.get("sensors", 0) if vns_data else 0)
    
    # Filtrer les cas où les deux sont à 0 (pas de données)
    filtered_data = [(inst, gv, vv) for inst, gv, vv in zip(instances, grasp_values, vns_values) if gv > 0 or vv > 0]
    if not filtered_data:
        print("Aucune donnée à afficher dans le graphique.")
        return
    
    instances_filtered, grasp_values_filtered, vns_values_filtered = zip(*filtered_data)
    
    x = range(len(instances_filtered))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(12, len(instances_filtered) * 0.8), 8))
    
    bars1 = ax.bar([i - width/2 for i in x], grasp_values_filtered, width, label="GRASP", alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], vns_values_filtered, width, label="VNS", alpha=0.8)
    
    ax.set_xlabel("Instance et paire (Rcapt, Rcom)", fontsize=12)
    ax.set_ylabel("Nombre de capteurs", fontsize=12)
    ax.set_title("Comparaison GRASP vs VNS - Nombre de capteurs", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(instances_filtered, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
    
    # Note si VNS n'a pas de données
    if all(v == 0 for v in vns_values_filtered):
        ax.text(0.5, 0.95, "Note: Aucune donnée VNS disponible", 
                transform=ax.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Graphique sauvegardé: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_time_comparison(results: List[Dict], save_path: Optional[str] = None, show: bool = True):
    """Génère un graphique comparant les temps de résolution."""
    organized = organize_by_instance(results)
    
    instances = []
    grasp_times = []
    vns_times = []
    
    for instance in sorted(organized.keys()):
        for pair in sorted(organized[instance].keys()):
            instances.append(f"{instance}\nR=({pair[0]},{pair[1]})")
            
            grasp_data = organized[instance][pair].get("grasp", {})
            vns_data = organized[instance][pair].get("vns", {})
            
            grasp_times.append(grasp_data.get("time_s", 0) if grasp_data else 0)
            vns_times.append(vns_data.get("time_s", 0) if vns_data else 0)
    
    # Filtrer les cas où les deux sont à 0
    filtered_data = [(inst, gt, vt) for inst, gt, vt in zip(instances, grasp_times, vns_times) if gt > 0 or vt > 0]
    if not filtered_data:
        print("Aucune donnée de temps à afficher.")
        return
    
    instances_filtered, grasp_times_filtered, vns_times_filtered = zip(*filtered_data)
    
    x = range(len(instances_filtered))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(12, len(instances_filtered) * 0.8), 8))
    
    bars1 = ax.bar([i - width/2 for i in x], grasp_times_filtered, width, label="GRASP", alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], vns_times_filtered, width, label="VNS", alpha=0.8)
    
    ax.set_xlabel("Instance et paire (Rcapt, Rcom)", fontsize=12)
    ax.set_ylabel("Temps de résolution (secondes)", fontsize=12)
    ax.set_title("Comparaison GRASP vs VNS - Temps de résolution", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(instances_filtered, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Graphique sauvegardé: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_by_radius_pair(results: List[Dict], save_path: Optional[str] = None, show: bool = True):
    """Génère un graphique comparant les performances par paire (Rcapt, Rcom)."""
    pairs = sorted(set((r["rcapt"], r["rcom"]) for r in results))
    
    grasp_avg = []
    vns_avg = []
    pair_labels = []
    
    for pair in pairs:
        pair_results = [r for r in results if r["rcapt"] == pair[0] and r["rcom"] == pair[1]]
        grasp_pair = [r["sensors"] for r in pair_results if r["algo"] == "grasp"]
        vns_pair = [r["sensors"] for r in pair_results if r["algo"] == "vns"]
        
        pair_labels.append(f"R=({pair[0]},{pair[1]})")
        grasp_avg.append(statistics.mean(grasp_pair) if grasp_pair else 0)
        vns_avg.append(statistics.mean(vns_pair) if vns_pair else 0)
    
    # Filtrer les paires sans données
    filtered_data = [(label, ga, va) for label, ga, va in zip(pair_labels, grasp_avg, vns_avg) if ga > 0 or va > 0]
    if not filtered_data:
        print("Aucune donnée par rayon à afficher.")
        return
    
    pair_labels_filtered, grasp_avg_filtered, vns_avg_filtered = zip(*filtered_data)
    
    x = range(len(pair_labels_filtered))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar([i - width/2 for i in x], grasp_avg_filtered, width, label="GRASP", alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], vns_avg_filtered, width, label="VNS", alpha=0.8)
    
    ax.set_xlabel("Paire (Rcapt, Rcom)", fontsize=12)
    ax.set_ylabel("Nombre moyen de capteurs", fontsize=12)
    ax.set_title("Performance moyenne par paire de rayons", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels_filtered)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Graphique sauvegardé: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def generate_all_comparisons(csv_path: str, output_dir: str = "results/comparisons", show: bool = False):
    """Génère tous les tableaux et graphiques de comparaison."""
    results = load_results_csv(csv_path)
    
    if not results:
        print(f"Aucun résultat trouvé dans {csv_path}")
        return
    
    print(f"\nAnalyse de {len(results)} résultats...")
    
    # Tableaux
    print_summary_table(results, use_pandas=HAS_PANDAS)
    print_statistics(results)
    
    # Graphiques
    os.makedirs(output_dir, exist_ok=True)
    
    plot_comparison_bar_chart(
        results,
        save_path=os.path.join(output_dir, "comparison_sensors.png"),
        show=show
    )
    
    plot_time_comparison(
        results,
        save_path=os.path.join(output_dir, "comparison_time.png"),
        show=show
    )
    
    plot_by_radius_pair(
        results,
        save_path=os.path.join(output_dir, "comparison_by_radius.png"),
        show=show
    )
    
    print(f"\nTous les graphiques ont été sauvegardés dans: {output_dir}")

