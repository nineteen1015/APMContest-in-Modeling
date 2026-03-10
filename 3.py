import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
from datetime import datetime, timedelta
import statsmodels.api as sm

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class SemiconductorImpactAnalyzer:
    def __init__(self):
        self.results = {}
        self.data_sources = {}

    def create_enhanced_semiconductor_data(self):
        years = np.arange(2015, 2026)

        global_market = {
            'year': years,
            'total_sales': [335, 339, 412, 469, 468, 440, 412, 466, 574, 630, 701],
            'usa_share': [0.50, 0.49, 0.48, 0.47, 0.46, 0.46, 0.47, 0.48, 0.49, 0.504, 0.51],
            'china_share': [0.10, 0.11, 0.12, 0.13, 0.15, 0.16, 0.17, 0.18, 0.19, 0.211, 0.23],
            'korea_share': [0.16, 0.17, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.182, 0.18],
            'taiwan_share': [0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.11, 0.11, 0.12, 0.122, 0.12]
        }

        self.data_sources['global_semiconductor'] = pd.DataFrame(global_market)

        usa_semiconductor = {
            'segment': ['High-End Logic', 'Memory', 'Analog', 'Discrete', 'Optoelectronics', 'Sensors'],
            'global_market_share': [0.52, 0.28, 0.45, 0.35, 0.40, 0.38],
            'domestic_production_ratio': [0.12, 0.08, 0.25, 0.45, 0.35, 0.40],
            'import_dependency': [0.88, 0.92, 0.75, 0.55, 0.65, 0.60],
            'rd_intensity': [0.25, 0.18, 0.15, 0.08, 0.12, 0.10],
            'employment_multiplier': [8.5, 6.2, 5.8, 4.3, 5.1, 4.7],
            'strategic_importance': [9.5, 8.0, 7.0, 6.0, 7.5, 7.0]
        }

        self.data_sources['usa_semiconductor_detail'] = pd.DataFrame(usa_semiconductor)

    def build_multidimensional_impact_model(self, detail_data):
        dimensions = {
            'economic_efficiency': 0.25,
            'national_security': 0.30,
            'technological_leadership': 0.20,
            'supply_chain_resilience': 0.15,
            'global_competitiveness': 0.10
        }

        results = []
        for _, chip_type in detail_data.iterrows():
            base_tariff = 0.02
            new_tariff = 0.15

            cost_increase = chip_type['import_dependency'] * (new_tariff - base_tariff) * 100
            production_response = min(0.20, new_tariff * 0.8 * (1 - chip_type['domestic_production_ratio']))

            if 'High-End' in chip_type['segment']:
                market_impact = -0.10
            else:
                market_impact = 0.03 * production_response - 0.02 * cost_increase / 100

            economic_score = self._calculate_economic_score(cost_increase, production_response, market_impact)
            security_score = self._calculate_security_score(chip_type, production_response)
            technology_score = self._calculate_technology_score(chip_type, market_impact)
            resilience_score = self._calculate_resilience_score(chip_type, production_response)
            competitiveness_score = self._calculate_competitiveness_score(chip_type, market_impact, cost_increase)

            weighted_score = (
                    economic_score * dimensions['economic_efficiency'] +
                    security_score * dimensions['national_security'] +
                    technology_score * dimensions['technological_leadership'] +
                    resilience_score * dimensions['supply_chain_resilience'] +
                    competitiveness_score * dimensions['global_competitiveness']
            )

            results.append({
                'chip_type': chip_type['segment'],
                'cost_increase_pct': cost_increase,
                'production_growth_pct': production_response * 100,
                'market_share_change_pct': market_impact * 100,
                'economic_score': economic_score,
                'security_score': security_score,
                'technology_score': technology_score,
                'resilience_score': resilience_score,
                'competitiveness_score': competitiveness_score,
                'comprehensive_score': weighted_score,
                'strategic_priority': chip_type['strategic_importance']
            })

        return pd.DataFrame(results)

    def _calculate_economic_score(self, cost_increase, production_growth, market_impact):
        base_score = 80
        cost_penalty = min(30, cost_increase * 2)
        production_bonus = production_growth * 150
        market_penalty = abs(min(0, market_impact)) * 100
        score = base_score - cost_penalty + production_bonus - market_penalty
        return max(0, min(100, score))

    def _calculate_security_score(self, chip_type, production_growth):
        base_score = chip_type['strategic_importance'] * 8
        localization_bonus = production_growth * 40
        import_risk = chip_type['import_dependency'] * 30
        score = base_score + localization_bonus - import_risk
        return max(0, min(100, score))

    def _calculate_technology_score(self, chip_type, market_impact):
        base_score = chip_type['rd_intensity'] * 120
        market_effect = market_impact * 80
        score = base_score + market_effect
        return max(0, min(100, score))

    def _calculate_resilience_score(self, chip_type, production_growth):
        base_resilience = (1 - chip_type['import_dependency']) * 70
        production_contribution = production_growth * 60
        score = base_resilience + production_contribution
        return max(0, min(100, score))

    def _calculate_competitiveness_score(self, chip_type, market_impact, cost_increase):
        base_competitiveness = chip_type['global_market_share'] * 90
        cost_penalty = cost_increase * 0.8
        market_effect = market_impact * 70
        score = base_competitiveness - cost_penalty + market_effect
        return max(0, min(100, score))

    def create_comprehensive_semiconductor_analysis(self):
        print("\n" + "=" * 80)
        print("Advanced Semiconductor Industry Impact Analysis")
        print("=" * 80)

        self.create_enhanced_semiconductor_data()
        global_data = self.data_sources['global_semiconductor']
        detail_data = self.data_sources['usa_semiconductor_detail']

        impact_analysis = self.build_multidimensional_impact_model(detail_data)

        self.create_advanced_semiconductor_visualizations(impact_analysis, global_data, detail_data)

        policy_recommendations = self.generate_semiconductor_policy_recommendations(impact_analysis)

        self.results['semiconductor_analysis'] = impact_analysis
        self.results['policy_recommendations'] = policy_recommendations

        return impact_analysis

    def create_advanced_semiconductor_visualizations(self, impact_data, global_data, detail_data):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 1. Multidimensional Impact Radar Chart
        categories = ['Economic Efficiency', 'National Security', 'Technology Leadership',
                      'Supply Chain Resilience', 'Global Competitiveness']
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        ax_radar = fig.add_subplot(2, 2, 1, polar=True)

        for _, row in impact_data.iterrows():
            values = [
                row['economic_score'] / 100,
                row['security_score'] / 100,
                row['technology_score'] / 100,
                row['resilience_score'] / 100,
                row['competitiveness_score'] / 100
            ]
            values += values[:1]
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=row['chip_type'], alpha=0.7)
            ax_radar.fill(angles, values, alpha=0.1)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Multidimensional Impact Assessment', size=14, fontweight='bold', pad=20)
        ax_radar.legend(bbox_to_anchor=(1.2, 1.0))

        # 2. Global Market Share Trends
        years = global_data['year']
        countries = ['usa_share', 'china_share', 'korea_share', 'taiwan_share']
        country_names = ['USA', 'China', 'Korea', 'Taiwan']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for i, country in enumerate(countries):
            ax2.plot(years, global_data[country] * 100, marker='o', linewidth=2,
                     label=country_names[i], color=colors[i])

        ax2.set_title('Global Semiconductor Market Share Trends', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Market Share (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Impact Analysis Heatmap
        impact_metrics = ['cost_increase_pct', 'production_growth_pct', 'market_share_change_pct',
                          'comprehensive_score']
        metric_names = ['Cost Increase (%)', 'Production Growth (%)', 'Market Share Change (%)', 'Comprehensive Score']

        heatmap_data = impact_data[impact_metrics].T
        im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

        ax3.set_xticks(np.arange(len(impact_data)))
        ax3.set_xticklabels(impact_data['chip_type'], rotation=45, ha='right')
        ax3.set_yticks(np.arange(len(metric_names)))
        ax3.set_yticklabels(metric_names)
        ax3.set_title('Semiconductor Impact Analysis Heatmap', fontsize=14, fontweight='bold')

        for i in range(len(metric_names)):
            for j in range(len(impact_data)):
                text = ax3.text(j, i, f'{heatmap_data.iloc[i, j]:.1f}',
                                ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax3)

        # 4. Strategic Priority vs Comprehensive Score
        scatter = ax4.scatter(impact_data['strategic_priority'], impact_data['comprehensive_score'],
                              s=impact_data['production_growth_pct'] * 10,
                              c=impact_data['cost_increase_pct'], cmap='viridis', alpha=0.7)

        for i, row in impact_data.iterrows():
            ax4.annotate(row['chip_type'], (row['strategic_priority'], row['comprehensive_score']),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax4.set_xlabel('Strategic Priority')
        ax4.set_ylabel('Comprehensive Score')
        ax4.set_title('Strategic Priority vs Comprehensive Impact', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax4, label='Cost Increase (%)')

        plt.tight_layout()
        plt.savefig('semiconductor_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Additional specialized visualizations
        self.create_supply_chain_analysis(detail_data)
        self.create_policy_scenario_analysis()

    def create_supply_chain_analysis(self, detail_data):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Supply chain risk analysis
        risk_factors = {
            'chip_type': detail_data['segment'],
            'geographic_concentration': [0.85, 0.90, 0.75, 0.60, 0.70, 0.65],
            'single_source_dependency': [0.65, 0.70, 0.55, 0.40, 0.50, 0.45],
            'critical_material_risk': [0.80, 0.75, 0.60, 0.45, 0.55, 0.50],
        }

        risk_df = pd.DataFrame(risk_factors)
        risk_weights = [0.4, 0.3, 0.3]
        risk_df['comprehensive_risk'] = (
                                                risk_df['geographic_concentration'] * risk_weights[0] +
                                                risk_df['single_source_dependency'] * risk_weights[1] +
                                                risk_df['critical_material_risk'] * risk_weights[2]
                                        ) * 100

        x = np.arange(len(risk_df))
        width = 0.25

        ax1.bar(x - width, risk_df['geographic_concentration'] * 100, width,
                label='Geographic Concentration', alpha=0.7)
        ax1.bar(x, risk_df['single_source_dependency'] * 100, width,
                label='Single Source Dependency', alpha=0.7)
        ax1.bar(x + width, risk_df['critical_material_risk'] * 100, width,
                label='Critical Material Risk', alpha=0.7)

        ax1.set_xlabel('Chip Type')
        ax1.set_ylabel('Risk Score (%)')
        ax1.set_title('Supply Chain Risk Analysis', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(risk_df['chip_type'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Import dependency vs domestic production
        ax2.scatter(detail_data['import_dependency'] * 100, detail_data['domestic_production_ratio'] * 100,
                    s=detail_data['strategic_importance'] * 100, alpha=0.7,
                    c=detail_data['global_market_share'] * 100, cmap='plasma')

        for i, row in detail_data.iterrows():
            ax2.annotate(row['segment'], (row['import_dependency'] * 100, row['domestic_production_ratio'] * 100),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax2.set_xlabel('Import Dependency (%)')
        ax2.set_ylabel('Domestic Production Ratio (%)')
        ax2.set_title('Import Dependency vs Domestic Production', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.colorbar(ax2.collections[0], ax=ax2, label='Global Market Share (%)')

        plt.tight_layout()
        plt.savefig('semiconductor_supply_chain_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_policy_scenario_analysis(self):
        scenarios = {
            'Tariff Only': {'tariff': 0.20, 'subsidy': 0.05},
            'Subsidy Only': {'tariff': 0.02, 'subsidy': 0.15},
            'Balanced Approach': {'tariff': 0.12, 'subsidy': 0.10},
            'Status Quo': {'tariff': 0.02, 'subsidy': 0.05}
        }

        fig, ax = plt.subplots(figsize=(12, 8))

        scenario_results = {}
        for scenario, params in scenarios.items():
            production_impact = params['subsidy'] * 60 - params['tariff'] * 20
            cost_impact = params['tariff'] * 15 - params['subsidy'] * 8
            security_impact = params['subsidy'] * 25 + params['tariff'] * 10

            scenario_results[scenario] = {
                'production_impact': production_impact,
                'cost_impact': cost_impact,
                'security_impact': security_impact,
                'overall_score': production_impact * 0.4 + security_impact * 0.4 - cost_impact * 0.2
            }

        metrics = ['production_impact', 'cost_impact', 'security_impact', 'overall_score']
        metric_names = ['Production Impact', 'Cost Impact', 'Security Impact', 'Overall Score']

        x = np.arange(len(scenarios))
        width = 0.2

        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [scenario_results[scenario][metric] for scenario in scenarios]
            ax.bar(x + i * width, values, width, label=metric_name, alpha=0.7)

        ax.set_xlabel('Policy Scenario')
        ax.set_ylabel('Impact Score')
        ax.set_title('Policy Scenario Impact Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(scenarios.keys(), rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig('semiconductor_policy_scenarios.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_semiconductor_policy_recommendations(self, impact_data):
        recommendations = {
            'immediate_actions': [
                "Implement differentiated tariff protection for high-end logic and memory chips (20-25%)",
                "Accelerate CHIPS Act funding distribution, focusing on sub-3nm advanced processes",
                "Establish strategic reserves for critical semiconductor materials"
            ],
            'medium_term_strategies': [
                "Build 'tariff-subsidy-tax' policy portfolio to balance protection and efficiency",
                "Promote semiconductor talent development, adding 10,000 professionals annually",
                "Strengthen US-EU-Japan-Korea semiconductor alliance to diversify supply chain risks"
            ],
            'long_term_visions': [
                "Achieve 25% localization rate for high-end chip manufacturing by 2030",
                "Establish complete semiconductor industry innovation ecosystem",
                "Maintain leadership in future fields like quantum computing and AI chips"
            ]
        }

        # Create recommendations visualization
        fig, ax = plt.subplots(figsize=(14, 8))

        categories = list(recommendations.keys())
        colors = ['#E74C3C', '#3498DB', '#2ECC71']

        y_pos = 0
        for i, (category, actions) in enumerate(recommendations.items()):
            for j, action in enumerate(actions):
                ax.barh(y_pos, len(action) / 10, height=0.6, color=colors[i], alpha=0.7)
                ax.text(0, y_pos, action, ha='left', va='center', fontsize=9,
                        style='italic', weight='bold')
                y_pos += 1
            y_pos += 0.5

        ax.set_yticks([])
        ax.set_xlabel('Recommendation Complexity Score')
        ax.set_title('Semiconductor Industry Policy Recommendations', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')

        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], alpha=0.7, label='Immediate Actions'),
            Patch(facecolor=colors[1], alpha=0.7, label='Medium-term Strategies'),
            Patch(facecolor=colors[2], alpha=0.7, label='Long-term Visions')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig('semiconductor_policy_recommendations.png', dpi=300, bbox_inches='tight')
        plt.show()

        return recommendations

    def generate_comprehensive_report(self):
        print("\n" + "=" * 100)
        print("US Tariff Policy Impact Advanced Analysis Report")
        print("=" * 100)

        print("\nExecuting comprehensive analysis...")
        semi_results = self.create_comprehensive_semiconductor_analysis()

        print("\n" + "=" * 80)
        print("Key Findings and Policy Implications")
        print("=" * 80)

        if 'semiconductor_analysis' in self.results:
            impact_data = self.results['semiconductor_analysis']

            print("\nSemiconductor Industry Deep Analysis Results:")
            print("-" * 50)

            for _, chip in impact_data.iterrows():
                print(f"\n{chip['chip_type']}:")
                print(f"  • Comprehensive Score: {chip['comprehensive_score']:.1f}/100")
                print(
                    f"  • Economic Efficiency: {chip['economic_score']:.1f} | National Security: {chip['security_score']:.1f}")
                print(
                    f"  • Cost Increase: {chip['cost_increase_pct']:.1f}% | Production Growth: {chip['production_growth_pct']:.1f}%")

        print(f"\nAnalysis Methodology:")
        print("-" * 50)
        print(
            "• Multi-dimensional assessment framework: Economic efficiency, national security, technology leadership, supply chain resilience, global competitiveness")
        print("• Dynamic scenario analysis: Multiple policy combination scenarios")
        print("• Supply chain risk assessment: Geographic concentration, single source dependency, critical materials")
        print(
            "• Technology development path: Covering sub-3nm processes, Chiplet, quantum computing and other cutting-edge technologies")

        print(f"\nReport Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)


def main():
    analyzer = SemiconductorImpactAnalyzer()
    analyzer.generate_comprehensive_report()

    with pd.ExcelWriter('semiconductor_analysis_results.xlsx') as writer:
        if 'semiconductor_analysis' in analyzer.results:
            analyzer.results['semiconductor_analysis'].to_excel(writer, sheet_name='Impact Analysis', index=False)

    print("\nAnalysis completed! Detailed results saved to Excel file.")


if __name__ == "__main__":
    main()