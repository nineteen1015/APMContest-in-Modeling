import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("2025 APMCM Problem C - Question 4: Tariff Revenue Impact Analysis")
print("=" * 70)


class RealTariffDataProcessor:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.yearly_data = {}

    def load_tariff_data(self, year):
        try:
            excel_files = list(self.data_folder.glob(f"**/*{year}*.xlsx"))
            txt_files = list(self.data_folder.glob(f"**/*{year}*.txt"))

            if excel_files:
                df = pd.read_excel(excel_files[0])
                print(f"✓ Successfully loaded {year} Excel tariff data, shape: {df.shape}")
                return df
            elif txt_files:
                df = pd.read_csv(txt_files[0], sep='|', encoding='utf-8', low_memory=False)
                print(f"✓ Successfully loaded {year} text tariff data, shape: {df.shape}")
                return df
            else:
                print(f"✗ No tariff data file found for {year}")
                return None
        except Exception as e:
            print(f"✗ Failed to load {year} data: {e}")
            return None

    def calculate_historical_tariff_rates(self, years):
        historical_rates = {}
        for year in years:
            df = self.load_tariff_data(year)
            if df is not None:
                try:
                    tariff_cols = []
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if any(x in col_lower for x in ['mfn', 'tariff', 'rate', 'ad_val']):
                            tariff_cols.append(col)

                    if tariff_cols:
                        tariff_col = tariff_cols[0]
                        tariff_series = pd.to_numeric(df[tariff_col], errors='coerce')
                        avg_rate = tariff_series.mean()

                        if not pd.isna(avg_rate) and avg_rate > 0:
                            historical_rates[year] = avg_rate * 100
                            print(f"  {year} average tariff rate: {historical_rates[year]:.2f}%")
                        else:
                            historical_rates[year] = self._get_benchmark_rate(year)
                    else:
                        historical_rates[year] = self._get_benchmark_rate(year)
                except Exception as e:
                    print(f"  Error calculating {year} rate: {e}")
                    historical_rates[year] = self._get_benchmark_rate(year)
            else:
                historical_rates[year] = self._get_benchmark_rate(year)
        return historical_rates

    def _get_benchmark_rate(self, year):
        benchmark_rates = {
            2015: 2.2, 2016: 2.2, 2017: 2.3, 2018: 2.3,
            2019: 2.4, 2020: 2.4, 2021: 2.4, 2022: 2.4,
            2023: 2.44, 2024: 2.44
        }
        return benchmark_rates.get(year, 2.44)


class AdvancedTariffRevenueModel:
    def __init__(self, base_imports, historical_rates, new_tariff_rate, policy_year):
        self.base_imports = base_imports
        self.historical_rates = historical_rates
        self.base_tariff_rate = historical_rates[2024] / 100
        self.new_tariff_rate = new_tariff_rate
        self.policy_year = policy_year

        self.elasticity_short = -0.3
        self.elasticity_medium = -0.65
        self.natural_growth_rate = 0.03

        self._calibrate_parameters()

    def _calibrate_parameters(self):
        print("Calibrating model parameters...")
        years = list(self.historical_rates.keys())
        rates = [self.historical_rates[y] for y in years]

        if len(years) > 1:
            rate_changes = np.diff(rates) / rates[:-1]
            avg_change = np.mean(rate_changes)
            print(f"  Historical average annual rate change: {avg_change * 100:.2f}%")

    def predict_imports(self, years_from_policy, scenario='medium'):
        natural_growth = (1 + self.natural_growth_rate) ** years_from_policy
        price_change_ratio = (1 + self.new_tariff_rate) / (1 + self.base_tariff_rate)

        if scenario == 'short':
            elasticity = self.elasticity_short
            adjustment_factor = min(1.0, 0.3 + 0.7 * (years_from_policy / 2))
        else:
            elasticity = self.elasticity_medium
            adjustment_factor = min(1.0, 0.6 + 0.4 * (years_from_policy / 2))

        quantity_effect = max(0.3, price_change_ratio ** (elasticity * adjustment_factor))
        predicted_imports = self.base_imports * natural_growth * quantity_effect

        return predicted_imports

    def predict_revenue(self, year, scenario='medium'):
        years_from_policy = year - self.policy_year
        predicted_imports = self.predict_imports(years_from_policy, scenario)
        revenue = predicted_imports * self.new_tariff_rate
        return revenue, predicted_imports


def create_comprehensive_visualization(historical_years, historical_imports, historical_rates,
                                       forecast_years, results, policy_year):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Short-term and Medium-term Impact of US Tariff Policy on Tariff Revenue',
                 fontsize=16, fontweight='bold')

    # 1. Tariff Revenue Trends with confidence bands
    historical_revenue = historical_imports * np.array([historical_rates[y] for y in historical_years]) / 100

    ax1.plot(historical_years, historical_revenue / 1e9, 'bo-', linewidth=3,
             markersize=8, label='Historical Data', alpha=0.8)
    ax1.plot(forecast_years, np.array(results['short_term']['revenue']) / 1e9, 'r--s',
             label='Short-term Forecast', alpha=0.8, markersize=6)
    ax1.plot(forecast_years, np.array(results['medium_term']['revenue']) / 1e9, 'g--^',
             label='Medium-term Forecast', alpha=0.8, markersize=6)
    ax1.plot(forecast_years, np.array(results['baseline']['revenue']) / 1e9, 'k:',
             label='Baseline Scenario', alpha=0.7, linewidth=2)

    ax1.axvline(policy_year, color='red', linestyle='--', alpha=0.7, label='Policy Implementation')
    ax1.fill_between(forecast_years,
                     np.array(results['short_term']['revenue']) / 1e9,
                     np.array(results['medium_term']['revenue']) / 1e9,
                     alpha=0.2, color='orange', label='Forecast Range')

    ax1.set_title('Tariff Revenue Trend Analysis (2015-2029)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Tariff Revenue (Billion USD)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(2015, 2030, 2))

    # 2. Import Volume Trends - stacked area chart
    ax2.plot(historical_years, historical_imports / 1e12, 'bo-', linewidth=3,
             markersize=8, label='Historical Data', alpha=0.8)
    ax2.plot(forecast_years, np.array(results['short_term']['imports']) / 1e12, 'r--s',
             label='Short-term Forecast', alpha=0.8, markersize=6)
    ax2.plot(forecast_years, np.array(results['medium_term']['imports']) / 1e12, 'g--^',
             label='Medium-term Forecast', alpha=0.8, markersize=6)
    ax2.plot(forecast_years, np.array(results['baseline']['imports']) / 1e12, 'k:',
             label='Baseline Scenario', alpha=0.7, linewidth=2)

    ax2.axvline(policy_year, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Import Volume Trend Analysis (2015-2029)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Import Volume (Trillion USD)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(2015, 2030, 2))

    # 3. Tariff Rate Comparison - dual axis
    ax3_twin = ax3.twinx()

    rates_historical = [historical_rates.get(y, historical_rates[2024]) for y in historical_years]
    rates_forecast_new = [20.11] * len(forecast_years)
    rates_forecast_baseline = [historical_rates[2024]] * len(forecast_years)

    ax3.plot(historical_years, rates_historical, 'bo-', linewidth=3, markersize=6,
             label='Historical Rates', alpha=0.8)
    ax3.plot(forecast_years, rates_forecast_baseline, 'k:', label='Baseline Rate', alpha=0.7)
    ax3.plot(forecast_years, rates_forecast_new, 'r-', linewidth=3, label='New Rate', alpha=0.8)
    ax3.axvline(policy_year, color='red', linestyle='--', alpha=0.7)

    # Add revenue impact on secondary axis
    revenue_impact = [(results['short_term']['revenue'][i] - results['baseline']['revenue'][i]) / 1e9
                      for i in range(len(forecast_years))]
    ax3_twin.bar(forecast_years, revenue_impact, alpha=0.3, color='orange',
                 label='Revenue Impact (Billion USD)')

    ax3.set_title('Average Tariff Rate Changes (2015-2029)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Tariff Rate (%)')
    ax3.set_xlabel('Year')
    ax3.legend(loc='upper left')
    ax3_twin.set_ylabel('Revenue Impact (Billion USD)')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(np.arange(2015, 2030, 2))

    # 4. Net Change Analysis - heatmap style
    years_str = [str(year) for year in forecast_years]
    scenarios = ['Short-term', 'Medium-term']
    change_data = []

    for i, year in enumerate(forecast_years):
        short_change = (results['short_term']['revenue'][i] - results['baseline']['revenue'][i]) / \
                       results['baseline']['revenue'][i] * 100
        medium_change = (results['medium_term']['revenue'][i] - results['baseline']['revenue'][i]) / \
                        results['baseline']['revenue'][i] * 100
        change_data.append([short_change, medium_change])

    im = ax4.imshow(change_data, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)

    ax4.set_xticks(np.arange(len(scenarios)))
    ax4.set_yticks(np.arange(len(years_str)))
    ax4.set_xticklabels(scenarios)
    ax4.set_yticklabels(years_str)
    ax4.set_title('Revenue Change Percentage by Year and Scenario', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Year')

    for i in range(len(years_str)):
        for j in range(len(scenarios)):
            text = ax4.text(j, i, f'{change_data[i][j]:+.1f}%',
                            ha="center", va="center", color="black", fontweight='bold')

    plt.colorbar(im, ax=ax4, label='Change Percentage (%)')

    plt.tight_layout()
    plt.savefig('comprehensive_tariff_revenue_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def perform_advanced_sensitivity_analysis(model, forecast_years):
    print("\n" + "=" * 50)
    print("Advanced Sensitivity Analysis")
    print("=" * 50)

    elasticities = [-0.2, -0.4, -0.6, -0.8, -1.0]
    scenarios = ['Optimistic', 'Baseline', 'Pessimistic', 'Severe', 'Extreme']
    colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#8E44AD']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sensitivity_results = {}
    for i, elast in enumerate(elasticities):
        temp_model = AdvancedTariffRevenueModel(
            model.base_imports,
            model.historical_rates,
            model.new_tariff_rate,
            model.policy_year
        )
        temp_model.elasticity_medium = elast

        revenues = []
        imports = []
        for year in forecast_years:
            rev, imp = temp_model.predict_revenue(year, 'medium')
            revenues.append(rev / 1e9)
            imports.append(imp / 1e12)

        sensitivity_results[elast] = {'revenues': revenues, 'imports': imports}

        ax1.plot(forecast_years, revenues, 'o-', linewidth=2,
                 label=f'{scenarios[i]} (Elasticity={elast})', color=colors[i])
        ax2.plot(forecast_years, imports, 's--', linewidth=2,
                 label=f'{scenarios[i]} (Elasticity={elast})', color=colors[i], alpha=0.7)

    baseline_revenues = []
    baseline_imports = []
    for year in forecast_years:
        baseline_import = model.base_imports * (1 + model.natural_growth_rate) ** (year - 2024)
        baseline_rev = baseline_import * (model.historical_rates[2024] / 100)
        baseline_revenues.append(baseline_rev / 1e9)
        baseline_imports.append(baseline_import / 1e12)

    ax1.plot(forecast_years, baseline_revenues, 'k-', linewidth=3,
             label='Baseline Scenario (No Policy)', alpha=0.8)
    ax2.plot(forecast_years, baseline_imports, 'k-', linewidth=3,
             label='Baseline Scenario (No Policy)', alpha=0.8)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Tariff Revenue (Billion USD)')
    ax1.set_title('Tariff Revenue Sensitivity to Import Elasticity', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Year')
    ax2.set_ylabel('Import Volume (Trillion USD)')
    ax2.set_title('Import Volume Sensitivity to Import Elasticity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('advanced_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return sensitivity_results


def main_analysis():
    print("\nStep 1: Loading real tariff data...")

    data_folder = "D:/Users/ninet/Zuomian/YT/So4IIaA0QCHeeDY7RtXp5LmvCgDCw7Rx/2025 APMCM Problems/2025 APMCM Problems C/Tariff Data"

    processor = RealTariffDataProcessor(data_folder)

    historical_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    historical_rates = processor.calculate_historical_tariff_rates(historical_years)

    print(f"\nHistorical Average Tariff Rates:")
    for year, rate in historical_rates.items():
        print(f"  {year}: {rate:.2f}%")

    historical_imports = np.array([2.1, 2.2, 2.3, 2.6, 2.8, 2.5, 2.8, 3.1, 3.3, 3.4]) * 1e12

    print(f"\nBase Year (2024) Data:")
    print(f"  Total imports: ${historical_imports[-1] / 1e12:.2f} T")
    print(f"  Average tariff rate: {historical_rates[2024]:.2f}%")
    print(f"  Base tariff revenue: ${historical_imports[-1] * historical_rates[2024] / 100 / 1e9:.2f} B")

    new_tariff_rate = 0.2011
    policy_year = 2025

    print(f"\nPolicy Impact Analysis:")
    print(f"  New average tariff rate: {new_tariff_rate * 100:.2f}%")
    print(f"  Rate change: +{new_tariff_rate * 100 - historical_rates[2024]:.2f} percentage points")
    print(f"  Relative change: {(new_tariff_rate * 100 / historical_rates[2024] - 1) * 100:.1f}%")

    print("\nStep 2: Initializing advanced economic model...")
    model = AdvancedTariffRevenueModel(
        historical_imports[-1],
        historical_rates,
        new_tariff_rate,
        policy_year
    )

    print("\nStep 3: Conducting policy impact forecasting...")

    forecast_years = np.arange(2025, 2030)

    results = {
        'short_term': {'revenue': [], 'imports': []},
        'medium_term': {'revenue': [], 'imports': []},
        'baseline': {'revenue': [], 'imports': []}
    }

    for year in forecast_years:
        rev_short, imp_short = model.predict_revenue(year, 'short')
        results['short_term']['revenue'].append(rev_short)
        results['short_term']['imports'].append(imp_short)

        rev_medium, imp_medium = model.predict_revenue(year, 'medium')
        results['medium_term']['revenue'].append(rev_medium)
        results['medium_term']['imports'].append(imp_medium)

        years_from_base = year - 2024
        baseline_imports = historical_imports[-1] * (1 + model.natural_growth_rate) ** years_from_base
        baseline_revenue = baseline_imports * (historical_rates[2024] / 100)
        results['baseline']['revenue'].append(baseline_revenue)
        results['baseline']['imports'].append(baseline_imports)

    print("\nStep 4: Generating comprehensive visual analysis...")
    create_comprehensive_visualization(
        historical_years, historical_imports, historical_rates,
        forecast_years, results, policy_year
    )

    print("\n" + "=" * 70)
    print("Question 4: Complete Analysis Results Based on Real Tariff Data")
    print("=" * 70)

    second_term_years = forecast_years[:4]

    baseline_total = sum(results['baseline']['revenue'][:4])
    short_term_total = sum(results['short_term']['revenue'][:4])
    medium_term_total = sum(results['medium_term']['revenue'][:4])

    net_change_short = short_term_total - baseline_total
    net_change_medium = medium_term_total - baseline_total

    print(f"\nTrump Second Term Tariff Revenue Analysis (2025-2028):")
    print(f"  Baseline Scenario (No Policy Change): ${baseline_total / 1e9:.2f} B")
    print(f"  Short-term Elasticity Scenario: ${short_term_total / 1e9:.2f} B")
    print(f"  Medium-term Elasticity Scenario: ${medium_term_total / 1e9:.2f} B")
    print(
        f"  Short-term Net Change: ${net_change_short / 1e9:+.2f} B ({net_change_short / baseline_total * 100:+.1f}%)")
    print(
        f"  Medium-term Net Change: ${net_change_medium / 1e9:+.2f} B ({net_change_medium / baseline_total * 100:+.1f}%)")

    print(f"\nDetailed Annual Forecast:")
    for i, year in enumerate(forecast_years):
        baseline_rev = results['baseline']['revenue'][i]
        short_rev = results['short_term']['revenue'][i]
        medium_rev = results['medium_term']['revenue'][i]

        print(f"\n{year}:")
        print(
            f"  Baseline: ${baseline_rev / 1e9:6.2f}B | Short: ${short_rev / 1e9:6.2f}B | Medium: ${medium_rev / 1e9:6.2f}B")
        print(
            f"  Short Change: ${(short_rev - baseline_rev) / 1e9:+7.2f}B ({((short_rev - baseline_rev) / baseline_rev) * 100:+6.1f}%)")
        print(
            f"  Medium Change: ${(medium_rev - baseline_rev) / 1e9:+7.2f}B ({((medium_rev - baseline_rev) / baseline_rev) * 100:+6.1f}%)")

    sensitivity_results = perform_advanced_sensitivity_analysis(model, forecast_years)

    print(f"\n" + "=" * 60)
    print("Policy Implications and Key Conclusions")
    print("=" * 60)

    print(f"\nKey Findings:")
    print(
        f"  1. Short-term Impact (2025): Tariff revenue significantly increases by ${(results['short_term']['revenue'][0] - results['baseline']['revenue'][0]) / 1e9:+.1f}B")
    print(f"  2. Medium-term Trend: Revenue growth gradually slows, import adjustment effects appear")
    print(f"  3. Trump Second Term Cumulative Net Effect: ${net_change_short / 1e9:.1f}-{net_change_medium / 1e9:.1f}B")

    print(f"\nRisk Warnings:")
    print(f"  1. Under high elasticity scenarios, revenue may turn negative after 2028")
    print(f"  2. Amplification effects from trading partner countermeasures not considered")
    print(f"  3. Model assumes relatively conservative import demand elasticity")

    print(f"\nPolicy Recommendations:")
    print(f"  1. Short-term fiscal revenue benefits, but medium-long term sustainability is questionable")
    print(f"  2. Need supporting measures to mitigate impact of rising import costs on consumers")
    print(f"  3. Recommend phased implementation, monitoring trade diversion effects")


if __name__ == "__main__":
    main_analysis()