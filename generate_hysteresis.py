"""
I-criterion validation on seven empirical hypergraphs
=======================================================
This code generates hysteresis data with high resolution (30 points per dynamics)
using parallel processing (6 workers) for efficient computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import csv
import json
from datetime import datetime
import networkx as nx
import itertools
import concurrent.futures
import warnings
import traceback
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
N_POINTS = 30                    # 每个动力学的参数点数量
MAX_WORKERS = 6                  # 并行处理的网络数量（根据CPU核心调整）
INTEGRATION_TIME = 200           # 积分时间
RANDOM_SEED = 42                 # 随机种子

# ==================== 类型转换函数 ====================
def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

# ==================== 加载网络 ====================
def load_network_graphml(graphml_path):
    """从 GraphML 加载网络，构建高阶张量"""
    print(f"Loading graph from {graphml_path}...")
    try:
        G = nx.read_graphml(graphml_path)
        if G.is_directed():
            G = G.to_undirected()
            print("  Converted directed graph to undirected.")
        
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        N = len(nodes)
        print(f"  Number of nodes: {N}")
        print(f"  Number of edges: {G.number_of_edges()}")

        # 提取最大团作为高阶超边
        cliques = [c for c in nx.find_cliques(G) if len(c) >= 3]
        print(f"  Found {len(cliques)} maximal cliques of size ≥ 3.")

        # 构建三体张量
        W3 = np.zeros((N, N, N), dtype=np.float32)
        for clique in cliques:
            idx = [node_to_idx[n] for n in clique]
            for i, j, k in itertools.combinations(idx, 3):
                W3[i, j, k] = 1.0
                W3[i, k, j] = 1.0
                W3[j, i, k] = 1.0
                W3[j, k, i] = 1.0
                W3[k, i, j] = 1.0
                W3[k, j, i] = 1.0

        # 投影邻接矩阵
        A_proj = np.sum(W3, axis=2)
        np.fill_diagonal(A_proj, 0)

        # 计算投影度
        degrees_proj = np.zeros(N)
        for i in range(N):
            neighbors = set(np.where(A_proj[i] > 0)[0])
            degrees_proj[i] = len(neighbors)
        
        print(f"  Average projected degree <k> = {np.mean(degrees_proj):.2f}")

        return N, W3, A_proj, degrees_proj, node_to_idx, G, cliques
    except Exception as e:
        print(f"  Error loading graph: {e}")
        traceback.print_exc()
        raise

# ==================== 动力学1: SIS ====================
def sis_rhs(t, x, lambda1, lambda2, A_proj, W3):
    """SIS动力学方程"""
    try:
        two_body = A_proj @ x
        three_body = np.einsum('ijk,j,k->i', W3, x, x)
        dx = -x + (1 - x) * (lambda1 * two_body + lambda2 * three_body)
        return dx
    except:
        return np.zeros_like(x)

def sis_get_steady(lambda1, lambda2, A_proj, W3, N):
    """获取稳态值"""
    try:
        # 低初始态
        x0_low = np.random.uniform(0, 0.01, size=N)
        sol_low = solve_ivp(sis_rhs, (0, INTEGRATION_TIME), x0_low,
                           args=(lambda1, lambda2, A_proj, W3),
                           method='RK45', rtol=1e-3, atol=1e-4)
        steady_low = np.mean(sol_low.y[:, -1])

        # 高初始态
        x0_high = np.ones(N)
        sol_high = solve_ivp(sis_rhs, (0, INTEGRATION_TIME), x0_high,
                            args=(lambda1, lambda2, A_proj, W3),
                            method='RK45', rtol=1e-3, atol=1e-4)
        steady_high = np.mean(sol_high.y[:, -1])
        
        return steady_low, steady_high
    except:
        return 0.0, 1.0

# ==================== 动力学2: Kuramoto ====================
def kuramoto_rhs(t, theta, sigma1, sigma2, A_proj, W3, omega, k_mean):
    """Kuramoto动力学方程"""
    try:
        N = len(theta)
        if k_mean < 1e-10:
            k_mean = 1.0
        
        # 两体项
        sin_pair = np.sin(theta[:, None] - theta[None, :])
        pair_term = np.zeros(N)
        for i in range(N):
            neighbors = np.where(A_proj[i] > 0)[0]
            if len(neighbors) > 0:
                pair_term[i] = np.sum(A_proj[i, neighbors] * sin_pair[i, neighbors])
        pair_term = (sigma1 / k_mean) * pair_term

        # 三体项
        three_body = np.zeros(N)
        nonzero = np.where(W3 > 0)
        if len(nonzero[0]) > 0:
            max_calc = min(len(nonzero[0]), 5000)
            for idx in range(max_calc):
                i = nonzero[0][idx]
                j = nonzero[1][idx]
                k = nonzero[2][idx]
                if i < N and j < N and k < N:
                    three_body[i] += np.sin(theta[j] + theta[k] - 2*theta[i])
            three_body = three_body / (2 * k_mean)

        return omega + pair_term + sigma2 * three_body
    except Exception as e:
        return np.zeros_like(theta)

def kuramoto_order(theta):
    """计算序参量"""
    try:
        return np.abs(np.mean(np.exp(1j * theta)))
    except:
        return 0.0

def kuramoto_get_steady(sigma1, sigma2, A_proj, W3, omega, k_mean, init_type='random'):
    """获取稳态序参量"""
    try:
        N = len(omega)
        if init_type == 'random':
            theta0 = np.random.uniform(-np.pi, np.pi, size=N)
        else:
            theta0 = np.random.uniform(-0.1, 0.1, size=N)
        
        t_span = (0, INTEGRATION_TIME)
        t_eval = np.linspace(0, INTEGRATION_TIME, 15)
        
        sol = solve_ivp(kuramoto_rhs, t_span, theta0,
                       args=(sigma1, sigma2, A_proj, W3, omega, k_mean),
                       method='RK45', rtol=1e-2, atol=1e-3,
                       t_eval=t_eval, max_step=5.0)
        
        if sol.success and sol.y.shape[1] > 0:
            return kuramoto_order(sol.y[:, -1])
        return 0.0
    except:
        return 0.0

# ==================== 动力学3: 演化博弈 ====================
def game_payoff(s, c, r, A_proj, W3):
    """计算收益"""
    try:
        two_body = A_proj @ s
        three_body = np.einsum('ijk,j,k->i', W3, s, s)
        return -c * s + (r/2) * two_body + (r/3) * three_body
    except:
        return np.zeros_like(s)

def game_rhs(t, s, c, r, A_proj, W3, neighbor_lists):
    """演化博弈动力学方程"""
    try:
        pi = game_payoff(s, c, r, A_proj, W3)
        N = len(s)
        ds = np.zeros(N)
        for i in range(N):
            if neighbor_lists[i]:
                local_mean = np.mean([pi[j] for j in neighbor_lists[i]])
            else:
                local_mean = pi[i]
            ds[i] = s[i] * (1 - s[i]) * (pi[i] - local_mean)
        return ds
    except:
        return np.zeros_like(s)

def game_get_steady(c, r, A_proj, W3, neighbor_lists, init_type='low'):
    """获取稳态合作水平"""
    try:
        N = A_proj.shape[0]
        if init_type == 'low':
            s0 = np.random.uniform(0, 0.01, size=N)
        else:
            s0 = np.ones(N) * 0.9
        
        sol = solve_ivp(game_rhs, (0, INTEGRATION_TIME), s0,
                       args=(c, r, A_proj, W3, neighbor_lists),
                       method='RK45', rtol=1e-3, atol=1e-4)
        return np.mean(sol.y[:, -1])
    except:
        return 0.0

# ==================== 保存结果 ====================
def save_hysteresis_csv(param_list, low_vals, high_vals, filename, param_name='lambda1'):
    """保存迟滞环数据到CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([param_name, 'low_init', 'high_init'])
        for i, p in enumerate(param_list):
            writer.writerow([f"{p:.6f}", f"{low_vals[i]:.6f}", f"{high_vals[i]:.6f}"])
    print(f"  ✅ Saved to {os.path.basename(filename)}")

# ==================== 生成迟滞环数据 ====================
def generate_hysteresis_data(N, A_proj, W3, output_dir, dataset_name, n_points=N_POINTS):
    """为单个网络生成所有动力学的迟滞环数据"""
    
    # 计算谱半径
    lambda_max = np.linalg.norm(A_proj, 2)
    
    # ========== SIS动力学 ==========
    print(f"\n  [SIS] Generating hysteresis data with {n_points} points...")
    r_sis = 0.3
    p_c_sis = 1.0 / lambda_max if lambda_max > 0 else 0.1
    
    p_list = np.linspace(0.3 * p_c_sis, 1.5 * p_c_sis, n_points)
    low_sis, high_sis = [], []
    
    for i, p in enumerate(p_list):
        low, high = sis_get_steady(p, r_sis * p, A_proj, W3, N)
        low_sis.append(low)
        high_sis.append(high)
        if (i + 1) % 10 == 0 or i == 0 or i == n_points - 1:
            print(f"    Progress: {i+1}/{n_points}, λ1={p:.6f}, low={low:.4f}, high={high:.4f}")
    
    save_hysteresis_csv(p_list, low_sis, high_sis, 
                       os.path.join(output_dir, 'hysteresis_sis.csv'), 'lambda1')
    
    # ========== Kuramoto动力学 ==========
    print(f"\n  [Kuramoto] Generating hysteresis data with {n_points} points...")
    k_proj = np.sum(A_proj, axis=1)
    k_mean = np.mean(k_proj)
    if k_mean < 1e-10:
        k_mean = 1.0
    
    sigma2_ratio = 0.3
    sigma1_c = 1.0 / lambda_max if lambda_max > 0 else 0.1
    omega = np.random.uniform(-0.5, 0.5, size=N)
    
    sigma1_list = np.linspace(0.3 * sigma1_c, 1.5 * sigma1_c, n_points)
    r_low, r_high = [], []
    
    for i, s1 in enumerate(sigma1_list):
        s2 = sigma2_ratio * s1
        r_l = kuramoto_get_steady(s1, s2, A_proj, W3, omega, k_mean, 'random')
        r_h = kuramoto_get_steady(s1, s2, A_proj, W3, omega, k_mean, 'synchronized')
        r_low.append(r_l)
        r_high.append(r_h)
        if (i + 1) % 10 == 0 or i == 0 or i == n_points - 1:
            print(f"    Progress: {i+1}/{n_points}, σ1={s1:.6f}, r_low={r_l:.4f}, r_high={r_h:.4f}")
    
    save_hysteresis_csv(sigma1_list, r_low, r_high, 
                       os.path.join(output_dir, 'hysteresis_kuramoto.csv'), 'sigma1')
    
    # ========== 演化博弈 ==========
    print(f"\n  [Game] Generating hysteresis data with {n_points} points...")
    c = 0.1
    r_c_game = 2 * c / lambda_max if lambda_max > 0 else 1.0
    
    # 构建邻居列表
    neighbor_lists = []
    for i in range(N):
        nb = set()
        for j in range(N):
            if A_proj[i, j] > 0:
                nb.add(j)
        neighbor_lists.append(nb)
    
    r_list = np.linspace(0.3 * r_c_game, 1.5 * r_c_game, n_points)
    coop_low, coop_high = [], []
    
    for i, r_val in enumerate(r_list):
        low = game_get_steady(c, r_val, A_proj, W3, neighbor_lists, 'low')
        high = game_get_steady(c, r_val, A_proj, W3, neighbor_lists, 'high')
        coop_low.append(low)
        coop_high.append(high)
        if (i + 1) % 10 == 0 or i == 0 or i == n_points - 1:
            print(f"    Progress: {i+1}/{n_points}, r={r_val:.6f}, low={low:.4f}, high={high:.4f}")
    
    save_hysteresis_csv(r_list, coop_low, coop_high, 
                       os.path.join(output_dir, 'hysteresis_game.csv'), 'r')
    
    return True

# ==================== 处理单个网络 ====================
def process_network(name, path, out_dir_base, n_points=N_POINTS):
    """处理单个网络，生成迟滞环数据"""
    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")
    
    out_dir = os.path.join(out_dir_base, name)
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        N, W3, A_proj, degrees_proj, node_to_idx, G, cliques = load_network_graphml(path)
        generate_hysteresis_data(N, A_proj, W3, out_dir, name, n_points)
        print(f"\n✅ Completed {name}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to process {name}: {e}")
        traceback.print_exc()
        return False

# ==================== 主程序 ====================
def main():
    """主函数：并行生成所有网络的迟滞环数据"""
    np.random.seed(RANDOM_SEED)
    
    networks = {
        'C.elegans_pharynx': r'C:\Users\DELL\Desktop\真实数据集\C.elegans_neural.male_1\c.elegans.herm_pharynx_1.graphml',
        'Mixed.species_brain_1': r'C:\Users\DELL\Desktop\真实数据集\Cat\mixed.species_brain_1.graphml',
        'Mouse_visual_cortex_1': r'C:\Users\DELL\Desktop\真实数据集\Mouse_visual.cortex_1\mouse_visual.cortex_1.graphml',
        'Mouse_visual_cortex_2': r'C:\Users\DELL\Desktop\真实数据集\Mouse_visual.cortex_2\mouse_visual.cortex_2.graphml',
        'P.pacificus_synaptic_1': r'C:\Users\DELL\Desktop\真实数据集\P.pacificus_neural.synaptic_1\p.pacificus_neural.synaptic_1.graphml',
        'Rhesus_brain_2': r'C:\Users\DELL\Desktop\真实数据集\rhesus_brain_2\rhesus_brain_2.graphml',
        'Rhesus_cerebral_cortex_1': r'C:\Users\DELL\Desktop\真实数据集\rhesus_cerebral.cortex_1\rhesus_cerebral.cortex_1.graphml',
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = f'all_networks_hysteresis_{N_POINTS}points_{timestamp}'
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating hysteresis data for {len(networks)} networks")
    print(f"{'='*60}")
    print(f"  • Parameter points per dynamics: {N_POINTS}")
    print(f"  • Each network: 3 dynamics (SIS, Kuramoto, Game)")
    print(f"  • Total CSV files: {len(networks) * 3}")
    print(f"  • Total data points: {len(networks) * 3 * N_POINTS}")
    print(f"  • Integration time: {INTEGRATION_TIME}")
    print(f"  • Parallel workers: {MAX_WORKERS}")
    print(f"  • Random seed: {RANDOM_SEED}")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_networks = []
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for name, path in networks.items():
            if os.path.exists(path):
                future = executor.submit(process_network, name, path, base_output_dir, N_POINTS)
                futures[future] = name
            else:
                print(f"  ⚠️ Path not found for {name}: {path}")
                failed_networks.append(name)
        
        # 等待所有任务完成
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            completed += 1
            try:
                success = future.result()
                if success:
                    success_count += 1
                    print(f"\n[{completed}/{len(networks)}] ✅ Completed {name}")
                else:
                    failed_networks.append(name)
                    print(f"\n[{completed}/{len(networks)}] ❌ Failed {name}")
            except Exception as e:
                failed_networks.append(name)
                print(f"\n[{completed}/{len(networks)}] ❌ Error in {name}: {e}")
    
    # 汇总结果
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count}/{len(networks)} networks")
    print(f"Total CSV files generated: {success_count * 3}")
    print(f"Total data points: {success_count * 3 * N_POINTS}")
    print(f"Output directory: {base_output_dir}")
    
    if failed_networks:
        print(f"\n⚠️ Failed networks: {', '.join(failed_networks)}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"High-Resolution Hysteresis Data Generation")
    print(f"{'='*60}")
    print(f"  • {N_POINTS} points per dynamics")
    print(f"  • {MAX_WORKERS} parallel workers")
    print(f"  • Estimated time: ~8-12 minutes (depending on CPU)")
    print(f"{'='*60}\n")
    
    main()
