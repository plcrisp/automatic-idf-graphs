"""
Implementação do Modelo de Bartlett-Lewis para desagregação estocástica de precipitação.

Este módulo implementa o modelo estocástico de Bartlett-Lewis para simular a estrutura
temporal da precipitação e realizar a desagregação de séries de precipitação de resolução
temporal grossa (como diária ou horária) em séries mais finas (como 10 minutos).

Parâmetros do Modelo (calibrados com base em série de alta resolução):
-----------------------------------------------------------------------
- lambda (λ): frequência média de ocorrência de tempestades (eventos de chuva) por dia
- beta (β): número médio de pulsos (pequenos eventos de chuva) por tempestade
- gamma (γ): taxa de término da tempestade (inverso da duração média do evento)
- eta (η): taxa de término de um pulso (inverso da duração média do pulso)
- mu (μ): intensidade média de precipitação por pulso (mm)

Fluxo de Uso:
-------------
1. **Calibração do Modelo**
   - Carregar série de precipitação de resolução fina com intervalo fixo (ex: 10 min)
   - Identificar eventos de chuva baseado em período seco mínimo entre eles
   - Calibrar os parâmetros do modelo baseado nas estatísticas desses eventos

2. **Desagregação**
   - Carregar série de precipitação de resolução grossa (ex: horária)
   - Para cada valor observado na série grossa:
     * Simular um evento de chuva baseado nos parâmetros calibrados
     * Ajustar a intensidade de chuva simulada para corresponder ao volume observado
     * Dividir a chuva simulada nos intervalos finos desejados

3. **Exportação e Validação**
   - Exportar parâmetros calibrados em formato YAML para reutilização
   - Comparar a série desagregada com dados reais visualmente e estatisticamente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, expon
from typing import List, Tuple, Optional, Dict
import yaml
from pathlib import Path


class BartlettLewisModel:
    """
    Modelo de Bartlett-Lewis para simulação e desagregação estocástica de precipitação.
    
    Attributes:
        params (dict): Dicionário contendo os parâmetros do modelo (lambda, beta, gamma, eta, mu)
        calibrated (bool): Indica se o modelo foi calibrado
    """
    
    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Inicializa o modelo de Bartlett-Lewis.
        
        Args:
            params: Dicionário opcional com parâmetros do modelo
        """
        self.params = params
        self.calibrated = bool(params)

    def load_and_preprocess_data(
        self,
        file_path: str,
        time_column: str,
        rainfall_column: str,
        interval_minutes: int = 10,
        fill_method: str = 'zero'
    ) -> pd.DataFrame:
        """
        Carrega e pré-processa dados de precipitação de um arquivo CSV.
        
        Args:
            file_path: Caminho para o arquivo CSV
            time_column: Nome da coluna de data/hora
            rainfall_column: Nome da coluna de precipitação
            interval_minutes: Intervalo de tempo em minutos
            fill_method: 'zero' para preencher faltantes com 0, 'drop' para remover
            
        Returns:
            DataFrame com dados de precipitação pré-processados
        """
        df = pd.read_csv(file_path, parse_dates=[time_column])
        df = df.set_index(time_column)[[rainfall_column]]
        df.columns = ['rainfall_mm']

        if fill_method == 'zero':
            df = df.asfreq(f'{interval_minutes}min', fill_value=0)
        elif fill_method == 'drop':
            df = df.asfreq(f'{interval_minutes}min').dropna()
        else:
            raise ValueError("fill_method deve ser 'zero' ou 'drop'")

        return df

    def identify_events(
        self,
        rainfall_series: pd.Series,
        inter_event_gap_minutes: int = 30
    ) -> List[pd.DataFrame]:
        """
        Identifica eventos de precipitação baseado em um período seco mínimo entre eventos.
        
        Args:
            rainfall_series: Série temporal de precipitação
            inter_event_gap_minutes: Período seco mínimo entre eventos (minutos)
            
        Returns:
            Lista de DataFrames, cada um representando um evento de chuva
        """
        events = []
        interval_minutes = (rainfall_series.index[1] - rainfall_series.index[0]).total_seconds() / 60
        gap_intervals = int(inter_event_gap_minutes / interval_minutes)
        
        # Adiciona padding de zeros no início e fim para garantir identificação correta
        padded_series = pd.concat([
            pd.Series([0] * gap_intervals, index=pd.date_range(
                start=rainfall_series.index[0] - pd.Timedelta(minutes=gap_intervals * interval_minutes),
                periods=gap_intervals, freq=f'{int(interval_minutes)}min')),
            rainfall_series,
            pd.Series([0] * gap_intervals, index=pd.date_range(
                start=rainfall_series.index[-1] + pd.Timedelta(minutes=interval_minutes),
                periods=gap_intervals, freq=f'{int(interval_minutes)}min'))
        ])
        
        in_event = False
        dry_spell_counter = 0
        event_start_idx = -1

        for i, value in enumerate(padded_series):
            if value > 0:
                if not in_event:
                    in_event = True
                    event_start_idx = i
                dry_spell_counter = 0
            else:
                dry_spell_counter += 1
                if in_event and dry_spell_counter >= gap_intervals:
                    event_end_idx = i - gap_intervals
                    event = padded_series.iloc[event_start_idx:event_end_idx + 1].loc[lambda x: x > 0]
                    if not event.empty and event.sum() > 0:
                        events.append(event.to_frame(name='rainfall_mm'))
                    in_event = False
                    dry_spell_counter = 0

        return events

    def extract_beta_eta(
        self,
        events: List[pd.DataFrame],
        interval_minutes: int = 10,
        intra_event_gap_minutes: int = 15
    ) -> Tuple[float, float]:
        """
        Estima os parâmetros beta e eta a partir de eventos de precipitação.
        
        Esta função analisa períodos secos dentro de cada evento para identificar pulsos
        e calcular:
        - beta: número médio de pulsos por evento
        - eta: inverso da duração média dos pulsos (em minutos)
        
        Args:
            events: Lista de eventos de precipitação
            interval_minutes: Resolução temporal da série de precipitação
            intra_event_gap_minutes: Período seco mínimo que separa pulsos dentro de um evento
            
        Returns:
            Tupla (beta, eta)
        """
        all_pulses_count = []
        all_pulse_durations = []

        gap_intervals = int(intra_event_gap_minutes / interval_minutes)

        for event in events:
            values = event['rainfall_mm'].values

            pulse_count = 0
            pulse_lengths = []

            i = 0
            while i < len(values):
                if values[i] > 0:
                    pulse_length = 1
                    zero_counter = 0
                    i += 1
                    while i < len(values):
                        if values[i] > 0:
                            pulse_length += 1
                            zero_counter = 0
                        else:
                            zero_counter += 1
                            if zero_counter >= gap_intervals:
                                break
                            else:
                                pulse_length += 1
                        i += 1
                    pulse_count += 1
                    pulse_lengths.append(pulse_length * interval_minutes)
                else:
                    i += 1

            if pulse_count > 0:
                all_pulses_count.append(pulse_count)
                all_pulse_durations.extend(pulse_lengths)

        beta = np.mean(all_pulses_count) if all_pulses_count else 1.0
        mean_pulse_duration = (
            np.mean(all_pulse_durations) if all_pulse_durations else 10.0
        )
        eta = 1.0 / mean_pulse_duration if mean_pulse_duration > 0 else 0.1

        return beta, eta

    def calibrate(
        self,
        events: List[pd.DataFrame],
        interval_minutes: int = 10,
        default_beta: Optional[float] = None,
        default_eta: Optional[float] = None,
        intra_event_gap_minutes: int = 15,
    ) -> Dict[str, float]:
        """
        Calibra os parâmetros do modelo usando o Método dos Momentos.
        
        Args:
            events: Lista de eventos de precipitação
            interval_minutes: Intervalo de tempo em minutos
            default_beta: Se fornecido, usa este valor de beta ao invés de estimar
            default_eta: Se fornecido, usa este valor de eta ao invés de estimar
            intra_event_gap_minutes: Período seco mínimo usado ao estimar beta e eta
            
        Returns:
            Dicionário com parâmetros calibrados
            
        Raises:
            ValueError: Se nenhum evento for encontrado para calibração
        """
        if not events:
            raise ValueError("Nenhum evento de precipitação encontrado para calibração.")

        durations = [len(e) * interval_minutes for e in events]
        intensities = [
            e['rainfall_mm'].sum() / (len(e) * interval_minutes) if len(e) > 0 else 0
            for e in events
        ]
        total_duration_minutes = (events[-1].index[-1] - events[0].index[0]).total_seconds() / 60 or 1

        lambda_param = len(events) / (total_duration_minutes / (24 * 60))
        gamma_param = 1.0 / np.mean(durations) if np.mean(durations) > 0 else 0.01
        mu_param = np.mean(intensities) if np.mean(intensities) > 0 else 0.1

        if default_beta is None or default_eta is None:
            beta_est, eta_est = self.extract_beta_eta(
                events, interval_minutes, intra_event_gap_minutes
            )
            beta_param = default_beta if default_beta is not None else beta_est
            eta_param = default_eta if default_eta is not None else eta_est
        else:
            beta_param = default_beta
            eta_param = default_eta

        self.params = {
            'lambda': lambda_param,
            'beta': beta_param,
            'gamma': gamma_param,
            'eta': eta_param,
            'mu': mu_param,
        }
        self.calibrated = True
        return self.params

    def generate_synthetic_rainfall(
        self,
        total_duration_minutes: int,
        output_interval_minutes: int = 10,
        initial_timestamp: pd.Timestamp = pd.Timestamp("2000-01-01"),
        seed: Optional[int] = None
    ) -> pd.Series:
        """
        Gera uma série temporal sintética de precipitação usando o modelo de Bartlett-Lewis.
        
        Args:
            total_duration_minutes: Duração total em minutos
            output_interval_minutes: Intervalo de saída em minutos
            seed: Semente aleatória opcional para reprodutibilidade
            
        Returns:
            Série temporal de precipitação sintética
            
        Raises:
            ValueError: Se o modelo não foi calibrado
        """
        if not self.calibrated:
            raise ValueError("O modelo deve ser calibrado primeiro.")
        if seed is not None:
            np.random.seed(seed)

        p = self.params
        n_intervals = total_duration_minutes // output_interval_minutes
        rainfall = pd.Series(
            0.0,
            index=pd.date_range(
                start=initial_timestamp,
                periods=n_intervals,
                freq=f'{output_interval_minutes}min'
            )
        )

        lambda_per_min = p['lambda'] / (24 * 60)
        n_storms = poisson.rvs(lambda_per_min * total_duration_minutes)

        for _ in range(n_storms):
            storm_start = np.random.uniform(0, total_duration_minutes)
            pulses = poisson.rvs(p['beta'])
            current_time = storm_start
            for _ in range(pulses):
                current_time += expon.rvs(scale=1 / p['gamma'])
                duration = expon.rvs(scale=1 / p['eta'])
                intensity = expon.rvs(scale=p['mu'])

                i_start = int(current_time // output_interval_minutes)
                i_end = int((current_time + duration) // output_interval_minutes)
                for i in range(i_start, i_end):
                    if 0 <= i < len(rainfall):
                        rainfall.iloc[i] += intensity

        return rainfall

    def disaggregate(
        self,
        coarse_series: pd.Series,
        fine_interval_minutes: int = 10,
        seed: Optional[int] = None
    ) -> pd.Series:
        """
        Desagrega uma série de precipitação grossa em uma resolução mais fina usando o modelo.
        
        Args:
            coarse_series: Série de precipitação de resolução grossa
            fine_interval_minutes: Intervalo fino em minutos
            seed: Semente aleatória opcional
            
        Returns:
            Série de precipitação desagregada
            
        Raises:
            ValueError: Se o modelo não foi calibrado
        """
        if not self.calibrated:
            raise ValueError("O modelo deve ser calibrado primeiro.")
        disagg = pd.Series(dtype=float)

        if len(coarse_series) > 1:
            coarse_interval_minutes = (coarse_series.index[1] - coarse_series.index[0]).total_seconds() / 60
        else:
            coarse_interval_minutes = 60

        for ts, value in coarse_series.items():
            fine_times = pd.date_range(
                start=ts,
                periods=int(coarse_interval_minutes / fine_interval_minutes),
                freq=f'{fine_interval_minutes}min'
            )

            if value == 0:
                new_segment = pd.Series(0.0, index=fine_times)
            else:
                sim = self.generate_synthetic_rainfall(
                    int(coarse_interval_minutes),
                    output_interval_minutes=fine_interval_minutes,
                    seed=seed
                )
                if sim.sum() > 0:
                    sim *= (value / sim.sum())
                else:
                    sim[:] = value / len(sim)
                sim.index = fine_times
                new_segment = sim

            if disagg.empty:
                disagg = new_segment
            else:
                disagg = pd.concat([disagg, new_segment])

        return disagg

    def export_params(self, path: str = 'bartlett_lewis_params.yaml'):
        """
        Exporta os parâmetros calibrados para um arquivo YAML.
        
        Args:
            path: Caminho do arquivo de saída
            
        Raises:
            ValueError: Se não há parâmetros calibrados para exportar
        """
        if not self.params:
            raise ValueError("Nenhum parâmetro calibrado para exportar.")
        safe_params = {k: float(v) for k, v in self.params.items()}
        with open(path, 'w') as f:
            yaml.dump(safe_params, f)
        print(f"✅ Parâmetros exportados para: {path}")

    def load_params(self, path: str = 'bartlett_lewis_params.yaml'):
        """
        Carrega os parâmetros do modelo de um arquivo YAML.
        
        Args:
            path: Caminho para o arquivo YAML
        """
        with open(path, 'r') as f:
            self.params = yaml.safe_load(f)
            self.calibrated = True
        print(f"✅ Parâmetros carregados de: {path}")
        print(self.params)
    
    def plot_comparison(self, original: pd.Series, disaggregated: pd.Series, title: str = 'Comparison') -> None:
        """Plot line comparison between original and disaggregated series."""
        plt.figure(figsize=(12, 5))
        plt.plot(original.index, original.values, label='Original', alpha=0.7)
        plt.plot(disaggregated.index, disaggregated.values, label='Disaggregated', alpha=0.7, linestyle='--')
        plt.title(title)
        plt.ylabel("Rainfall (mm)")
        plt.xlabel("Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('rainfall_comparison.png')
        plt.show()

    def plot_comparison_bars(self, original: pd.Series, disaggregated: pd.Series, title: str = 'Comparison - Bars') -> None:
        """Plot bar comparison between original and disaggregated series."""
        plt.figure(figsize=(14, 6))
        bar_width = (original.index[1] - original.index[0]).total_seconds() / 60

        plt.bar(original.index, original.values, width=bar_width / 1440, label='Original', alpha=0.6, align='center')
        plt.bar(disaggregated.index, disaggregated.values, width=bar_width / 1440, label='Disaggregated', alpha=0.6, align='center')

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Rainfall (mm)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('rainfall_comparison_bars.png')
        plt.show()


def disaggregate_daily_to_subdaily(
    df_daily: pd.DataFrame,
    name_file: str,
    output_dir: str = 'Results',
    fine_interval_minutes: int = 10,
    calibration_file: Optional[str] = None,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Desagrega dados diários de precipitação em dados subdiários usando Bartlett-Lewis.
    
    Esta função integra-se com o fluxo do projeto idf_analysis para desagregar
    dados diários em intervalos finos (ex: 10 minutos) usando o modelo estocástico
    de Bartlett-Lewis.
    
    Args:
        df_daily: DataFrame com dados diários (deve conter coluna 'Precipitation')
        name_file: Nome base do arquivo de saída
        output_dir: Diretório de saída
        fine_interval_minutes: Intervalo fino desejado em minutos
        calibration_file: Caminho opcional para arquivo YAML com parâmetros calibrados
        seed: Semente aleatória para reprodutibilidade
        
    Returns:
        DataFrame com dados desagregados contendo máximos por intervalo
    """
    # Verifica se o DataFrame tem as colunas necessárias
    required_cols = ['Year', 'Precipitation']
    if not all(col in df_daily.columns for col in required_cols):
        raise ValueError(f"DataFrame deve conter as colunas: {required_cols}")
    
    # Criar índice temporal para a série diária
    if 'Month' in df_daily.columns and 'Day' in df_daily.columns:
        df_daily['Date'] = pd.to_datetime(df_daily[['Year', 'Month', 'Day']])
    else:
        # Assume primeiro dia do ano se não há informação de mês/dia
        df_daily['Date'] = pd.to_datetime(df_daily['Year'], format='%Y')
    
    df_daily = df_daily.set_index('Date')
    
    # Inicializar modelo
    bl_model = BartlettLewisModel()
    
    # Carregar parâmetros calibrados ou usar padrões
    if calibration_file and Path(calibration_file).exists():
        bl_model.load_params(calibration_file)
        print(f"✅ Usando parâmetros calibrados de: {calibration_file}")
    else:
        # Parâmetros padrão para desagregação de dados diários
        # IMPORTANTE: Estes parâmetros são ajustados para gerar intensidades
        # compatíveis com CETESB quando normalizados para volume diário total.
        # Para melhores resultados, calibre com dados de alta resolução!
        bl_model.params = {
            'lambda': 0.08,   # ~0.08 eventos/dia (1 evento a cada ~12 dias)
            'beta': 12.0,     # ~12 pulsos por evento (eventos mais estruturados)
            'gamma': 0.01,    # ~100 minutos por evento (eventos longos)
            'eta': 0.05,      # ~20 minutos por pulso
            'mu': 2.5,        # ~2.5 mm por pulso (alta intensidade quando ocorre)
        }
        bl_model.calibrated = True
        print("⚠️  Usando parâmetros padrão (recomenda-se FORTEMENTE calibração com dados locais)")
        print("   Estes parâmetros geram eventos esparsos mas intensos quando normalizados.")
    
    # Processar desagregação dia por dia
    all_disaggregated = []
    
    for idx, row in df_daily.iterrows():
        daily_precip = row['Precipitation']
        
        if daily_precip == 0:
            # Sem precipitação - gerar zeros
            n_intervals = int(1440 / fine_interval_minutes)  # 1440 min = 1 dia
            fine_series = pd.Series(0.0, index=range(n_intervals))
        else:
            # Gerar chuva sintética e normalizar para o total diário
            synthetic = bl_model.generate_synthetic_rainfall(
                total_duration_minutes=1440,  # 1 dia = 1440 minutos
                output_interval_minutes=fine_interval_minutes,
                seed=seed
            )
            
            # Normalizar para conservar volume total
            if synthetic.sum() > 0:
                synthetic *= (daily_precip / synthetic.sum())
            else:
                # Se não gerou chuva, distribuir uniformemente
                n_intervals = int(1440 / fine_interval_minutes)
                synthetic = pd.Series(daily_precip / n_intervals, index=range(n_intervals))
            
            fine_series = synthetic
        
        # Adicionar data ao índice
        fine_series.index = pd.date_range(
            start=idx,
            periods=len(fine_series),
            freq=f'{fine_interval_minutes}min'
        )
        all_disaggregated.append(fine_series)
    
    # Calcular máximos para intervalos compatíveis com o IDF
    intervals_minutes = [5, 10, 15, 20, 25, 30, 60, 180, 360, 480, 600, 720, 1440]
    
    # Criar DataFrame de resultados
    results_list = []
    
    # Processar cada dia individualmente para evitar cruzamento de dias
    for idx, row in df_daily.iterrows():
        year = row['Year']
        day_series = all_disaggregated[len(results_list)]  # Série deste dia
        
        result_row = {'Year': year}
        
        for interval_min in intervals_minutes:
            # Calcular rolling sum DENTRO do dia
            window_size = int(interval_min / fine_interval_minutes)
            
            if window_size > 0 and len(day_series) >= window_size:
                # Rolling sum dentro deste dia específico
                rolling_sum = day_series.rolling(window=window_size).sum()
                max_value = rolling_sum.max() if len(rolling_sum) > 0 else 0.0
            else:
                max_value = 0.0
            
            # Nome da coluna
            if interval_min < 60:
                col_name = f'Max_{interval_min}min'
            else:
                col_name = f'Max_{interval_min // 60}h'
            
            result_row[col_name] = round(max_value, 2)
        
        results_list.append(result_row)
    
    # Criar DataFrame final
    df_result = pd.DataFrame(results_list)
    
    # Salvar resultado
    output_path = Path(output_dir) / f"max_subdaily_{name_file}_bl.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_path, index=False)
    print(f"✅ Desagregação Bartlett-Lewis salva em: {output_path}")
    
    return df_result
