import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import pymannkendall as mk
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import warnings


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """Valida DataFrame e retorna versão limpa"""
    if df is None or df.empty:
        raise ValueError("DataFrame está vazio ou é None")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")
    
    df_clean = df.dropna(subset=required_columns)
    if df_clean.empty:
        raise ValueError("DataFrame não contém dados válidos após limpeza")
    
    return df_clean


def run_mann_kendall_test(data: pd.Series, test_type: str, alpha: float = 0.05) -> Dict:
    """Executa teste específico de Mann-Kendall"""
    test_functions = {
        'original': mk.original_test,
        'hamed_rao': mk.hamed_rao_modification_test,
        'yue_wang': mk.yue_wang_modification_test,
        'trend_free': mk.trend_free_pre_whitening_modification_test,
        'pre_whitening': mk.pre_whitening_modification_test
    }
    
    if test_type not in test_functions:
        raise ValueError(f"Teste '{test_type}' não disponível. Opções: {list(test_functions.keys())}")
    
    try:
        result = test_functions[test_type](data, alpha=alpha)
        return {
            'trend': result[0],
            'significant': result[1],
            'p_value': result[2],
            'z_score': result[3],
            'tau': result[4],
            's_statistic': result[5],
            'variance_s': result[6],
            'slope': result[7],
            'intercept': result[8]
        }
    except Exception as e:
        warnings.warn(f"Erro no teste {test_type}: {str(e)}")
        return {
            'trend': 'no trend', 'significant': False, 'p_value': np.nan,
            'z_score': np.nan, 'tau': np.nan, 's_statistic': np.nan,
            'variance_s': np.nan, 'slope': np.nan, 'intercept': np.nan
        }


def calculate_confidence_interval(data: pd.Series, slope: float, confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calcula intervalo de confiança para a tendência"""
    if np.isnan(slope) or len(data) < 3:
        return (np.nan, np.nan)
    
    std_error = np.std(data) / np.sqrt(len(data))
    margin = 1.96 * std_error  # Para 95% de confiança
    return (slope - margin, slope + margin)



def analyze_single_site(df: pd.DataFrame, site_name: str, 
                       precipitation_col: str = 'Precipitation',
                       time_col: str = 'Year',
                       alpha: float = 0.05) -> pd.DataFrame:
    """Analisa tendências para um único site"""
    df_clean = validate_dataframe(df, [precipitation_col, time_col])
    precipitation_data = df_clean[precipitation_col]
    
    test_types = ['original', 'hamed_rao', 'yue_wang', 'trend_free', 'pre_whitening']
    results = []
    
    for test_type in test_types:
        test_result = run_mann_kendall_test(precipitation_data, test_type, alpha)
        ci = calculate_confidence_interval(precipitation_data, test_result['slope'])
        
        results.append({
            'site': site_name,
            'test_type': test_type,
            'tau': test_result['tau'],
            'p_value': test_result['p_value'],
            'trend': test_result['trend'],
            'significant': test_result['significant'],
            'z_score': test_result['z_score'],
            's_statistic': test_result['s_statistic'],
            'variance_s': test_result['variance_s'],
            'slope': test_result['slope'],
            'intercept': test_result['intercept'],
            'ci_lower': ci[0],
            'ci_upper': ci[1],
        })
    
    return pd.DataFrame(results)


def analyze_multiple_sites(data_dict: Dict[str, pd.DataFrame],
                          precipitation_col: str = 'Precipitation',
                          time_col: str = 'Year',
                          alpha: float = 0.05) -> pd.DataFrame:
    """Analisa tendências para múltiplos sites"""
    all_results = []
    
    for site_name, df in data_dict.items():
        try:
            site_results = analyze_single_site(df, site_name, precipitation_col, time_col, alpha)
            all_results.append(site_results)
        except Exception as e:
            warnings.warn(f"Erro ao analisar {site_name}: {str(e)}")
            continue
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def plot_trend_analysis(df: pd.DataFrame, site_name: str, test_type: str,
                       slope: float, intercept: float, tau: float, p_value: float,
                       significant: bool, ci_lower: float, ci_upper: float, 
                       time_col: str = 'Year', precipitation_col: str = 'Precipitation',
                       output_dir: str = 'graphs') -> plt.Figure:
    """Gera gráfico de análise de tendência"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    x = df[time_col]
    y = df[precipitation_col]
    
    # Gráfico principal
    ax1.scatter(x, y, alpha=0.6, color='steelblue', s=50, label='Observações')
    
    # Linha de regressão linear
    try:
        slope_lr, intercept_lr, r_lr, _, _ = linregress(x, y)
        y_lr = slope_lr * x + intercept_lr
        ax1.plot(x, y_lr, color='green', linestyle='--', alpha=0.7, 
               label=f'Regressão Linear (R²={r_lr**2:.3f})')
    except:
        pass
    
    # Linha de tendência Mann-Kendall
    if not np.isnan(slope) and not np.isnan(intercept):
        y_mk = slope * x + intercept
        color = 'red' if significant else 'orange'
        linestyle = '-' if significant else ':'
        
        ax1.plot(x, y_mk, color=color, linestyle=linestyle, linewidth=2,
               label=f'Sen\'s Slope (τ={tau:.3f}, p={p_value:.3f})')
        
        # Intervalo de confiança
        if not np.isnan(ci_lower) and not np.isnan(ci_upper):
            y_ci_low = ci_lower * x + intercept
            y_ci_high = ci_upper * x + intercept
            ax1.fill_between(x, y_ci_low, y_ci_high, alpha=0.2, color=color, label='IC 95%')
    
    ax1.set_xlabel(time_col)
    ax1.set_ylabel(f'{precipitation_col} (mm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Informações do teste
    textstr = f'Tendência: {test_type.replace("_", " ").title()}\n'
    textstr += f'Significativo: {"Sim" if significant else "Não"}\n'
    textstr += f'p-valor: {p_value:.4f}\n'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Gráfico de resíduos
    if not np.isnan(slope) and not np.isnan(intercept):
        y_pred = slope * x + intercept
        residuals = y - y_pred
        ax2.scatter(x, residuals, alpha=0.6, color='purple', s=30)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        ax2.set_xlabel(time_col)
        ax2.set_ylabel('Resíduos (mm)')
        ax2.set_title('Análise de Resíduos')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Dados insuficientes para análise de resíduos',
               ha='center', va='center', transform=ax2.transAxes)
    
    plt.suptitle(f'Análise de Tendência - {site_name} ({test_type.replace("_", " ").title()})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{site_name}_{test_type}_trend_analysis.png"
    plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def plot_summary_comparison(results_df: pd.DataFrame, output_dir: str = 'graphs') -> plt.Figure:
    """Gera gráfico comparativo de resultados"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribuição de p-valores
    ax1 = axes[0, 0]
    ax1.hist(results_df['p_value'], bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
    ax1.set_xlabel('p-valor')
    ax1.set_ylabel('Frequência')
    ax1.set_title('Distribuição de p-valores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tau por tipo de teste
    ax2 = axes[0, 1]
    test_types = results_df['test_type'].unique()
    tau_by_test = [results_df[results_df['test_type'] == t]['tau'].values for t in test_types]
    ax2.boxplot(tau_by_test, labels=[t.replace('_', ' ').title() for t in test_types])
    ax2.set_ylabel('Tau de Kendall')
    ax2.set_title('Distribuição do Tau por Teste')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Slopes significativas
    ax3 = axes[1, 0]
    sig_results = results_df[results_df['significant']]
    if not sig_results.empty:
        ax3.hist(sig_results['slope'], bins=15, alpha=0.7, edgecolor='black', color='green')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.7)
        ax3.set_xlabel('Sen\'s Slope (mm/ano)')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição de Slopes Significativas')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Nenhuma tendência\nsignificativa encontrada',
               ha='center', va='center', transform=ax3.transAxes)
    
    # Proporção de tendências por site
    ax4 = axes[1, 1]
    trend_counts = results_df.groupby(['site', 'trend']).size().unstack(fill_value=0)
    if not trend_counts.empty:
        trend_counts.plot(kind='bar', stacked=True, ax=ax4, color=['red', 'gray', 'blue'])
        ax4.set_xlabel('Site')
        ax4.set_ylabel('Número de Testes')
        ax4.set_title('Distribuição de Tendências por Site')
        ax4.legend(title='Tendência')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'trend_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def filter_significant_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas resultados significativos"""
    return results_df[results_df['significant']].copy()


def get_best_test_per_site(results_df: pd.DataFrame) -> pd.DataFrame:
    """Retorna o melhor teste por site (menor p-valor entre os significativos)"""
    significant = filter_significant_results(results_df)
    if significant.empty:
        return pd.DataFrame()
    
    return significant.loc[significant.groupby('site')['p_value'].idxmin()].copy()


def generate_trend_plots(data_dict: Dict[str, pd.DataFrame], 
                        results_df: pd.DataFrame,
                        precipitation_col: str = 'Precipitation',
                        time_col: str = 'Year',
                        output_dir: str = 'graphs',
                        plot_all: bool = False) -> None:
    """Gera gráficos de tendência"""
    target_results = results_df if plot_all else filter_significant_results(results_df)
    
    for _, row in target_results.iterrows():
        site_name = row['site']
        if site_name in data_dict:
            plot_trend_analysis(
                data_dict[site_name], site_name, row['test_type'],
                row['slope'], row['intercept'], row['tau'], row['p_value'],
                row['significant'], row['ci_lower'], row['ci_upper'], 
                time_col, precipitation_col, output_dir
            )


def create_summary_stats(results_df: pd.DataFrame) -> Dict:
    """Cria estatísticas resumo da análise"""
    if results_df.empty:
        return {}
    
    significant_results = filter_significant_results(results_df)
    
    stats = {
        'total_tests': len(results_df),
        'total_sites': results_df['site'].nunique(),
        'significant_tests': len(significant_results),
        'significance_rate': len(significant_results) / len(results_df) * 100,
        'test_types': results_df['test_type'].unique().tolist()
    }
    
    if not significant_results.empty:
        increasing = len(significant_results[significant_results['trend'] == 'increasing'])
        decreasing = len(significant_results[significant_results['trend'] == 'decreasing'])
        
        stats.update({
            'increasing_trends': increasing,
            'decreasing_trends': decreasing,
            'mean_slope_significant': significant_results['slope'].mean(),
            'mean_tau_significant': significant_results['tau'].mean(),
        })
    
    return stats


def run_trend_analysis(data_dict: Dict[str, pd.DataFrame],
                      precipitation_col: str = 'Precipitation',
                      time_col: str = 'Year',
                      alpha: float = 0.05,
                      generate_plots: bool = False,
                      output_dir: str = 'trend_analysis_output') -> Tuple[pd.DataFrame, Dict]:
    """
    Executa análise completa de tendências
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dicionário com nome do site como chave e DataFrame como valor
    precipitation_col : str
        Nome da coluna de precipitação
    time_col : str
        Nome da coluna de tempo/ano
    alpha : float
        Nível de significância
    generate_plots : bool
        Se deve gerar gráficos
    output_dir : str
        Diretório de saída
    
    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        DataFrame com resultados e dicionário com estatísticas resumo
    """
    results_df = analyze_multiple_sites(data_dict, precipitation_col, time_col, alpha)
    stats = create_summary_stats(results_df)
    
    if generate_plots and not results_df.empty:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        generate_trend_plots(data_dict, results_df, precipitation_col, time_col, output_dir)
        plot_summary_comparison(results_df, output_dir)
        print(f'\n[INFO] Gráficos salvos em: {output_dir}\n')
    
    return results_df, stats