"""
Economic analysis visualizations.

This module provides interactive Plotly visualizations for economic decision-making:
- Cost breakdown (pie charts, waterfall charts)
- Savings vs baseline comparisons
- ROI and payback period displays
- Financial KPI dashboards
"""

from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flex_model.visualization.core.result_processor import OptimizationResult

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from flex_model.visualization.core.color_schemes import get_color_scheme


class EconomicPlots:
    """
    Factory class for economic analysis visualizations.

    All methods are static and return plotly.graph_objects.Figure instances.
    """

    @staticmethod
    def create_cost_breakdown(
        result: 'OptimizationResult',
        breakdown_type: str = 'by_asset',
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create cost breakdown visualization.

        Shows how total costs are distributed across assets and cost components.

        Args:
            result:
                OptimizationResult instance.

            breakdown_type:
                'by_asset' - Pie chart showing cost per asset (default)
                'by_component' - Pie chart showing cost by type (CAPEX, OPEX, market, etc.)

            template:
                Plotly template name ('plotly_white', 'plotly_dark', 'plotly', etc.)

        Returns:
            plotly.graph_objects.Figure with pie chart.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations. Install with: pip install plotly")

        # Get cost breakdown from result
        cost_data = result.get_cost_breakdown()

        if breakdown_type == 'by_asset':
            # Extract costs by asset
            labels = []
            values = []

            for asset_name, asset_costs in cost_data['by_asset'].items():
                labels.append(asset_name)
                values.append(abs(asset_costs.get('total', 0.0)))  # Use absolute value for pie chart

            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,  # Donut chart
                textinfo='label+percent',
                textposition='auto',
            )])

            fig.update_layout(
                title=f'Cost Breakdown by Asset<br><sub>Total: {cost_data["total_cost"]:.2f} CHF</sub>',
                template=template,
            )

        else:  # breakdown_type == 'by_component'
            # For now, use simplified component breakdown
            # Future: Extract detailed components from LP solution
            labels = ['Operational Cost']
            values = [abs(cost_data['total_cost'])]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                textinfo='label+value',
                textposition='auto',
            )])

            fig.update_layout(
                title='Cost Breakdown by Component',
                template=template,
            )

        return fig

    @staticmethod
    def create_savings_comparison(
        baseline_cost: float,
        optimized_cost: float,
        investment_cost: float = 0.0,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create savings comparison bar chart.

        Compares baseline (no FlexAssets) vs optimized (with FlexAssets) costs.

        Args:
            baseline_cost:
                Annual cost without FlexAssets [CHF/year].

            optimized_cost:
                Annual cost with FlexAssets [CHF/year].

            investment_cost:
                Upfront investment required [CHF]. If provided, shown as annotation.

        Returns:
            plotly.graph_objects.Figure with grouped bar chart.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        # Calculate savings
        absolute_savings = baseline_cost - optimized_cost
        relative_savings = (absolute_savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

        # Create bar chart
        fig = go.Figure()

        # Add bars
        fig.add_trace(go.Bar(
            x=['Baseline (No FlexAssets)', 'Optimized (With FlexAssets)'],
            y=[baseline_cost, optimized_cost],
            text=[f'{baseline_cost:,.0f} CHF', f'{optimized_cost:,.0f} CHF'],
            textposition='auto',
            marker_color=[colors.neutral_indicator_color, colors.positive_indicator_color],
            name='Annual Cost',
        ))

        # Add savings annotation
        annotations = [
            dict(
                x=1,
                y=max(baseline_cost, optimized_cost) * 0.5,
                text=f'<b>Savings</b><br>{absolute_savings:,.0f} CHF/year<br>({relative_savings:.1f}%)',
                showarrow=True,
                arrowhead=2,
                ax=80,
                ay=-40,
                font=dict(size=12, color="black"),
                bgcolor=colors.annotation_color,
                bordercolor='black',
                borderwidth=2,
            )
        ]

        # Add investment cost annotation if provided
        if investment_cost > 0:
            annotations.append(dict(
                x=1,
                y=optimized_cost,
                text=f'Investment: {investment_cost:,.0f} CHF',
                showarrow=True,
                arrowhead=2,
                ax=80,
                ay=40,
                font=dict(size=10, color=colors.gauge_medium_color),
            ))

        fig.update_layout(
            title='Cost Comparison: Baseline vs Optimized',
            yaxis_title='Annual Cost [CHF/year]',
            template=template,
            annotations=annotations,
            showlegend=False,
        )

        return fig

    @staticmethod
    def create_roi_gauge(
        roi: float,
        target_roi: float = 15.0,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create ROI gauge visualization.

        Visual indicator of return on investment with target benchmark.

        Args:
            roi:
                Return on investment [%].

            target_roi:
                Target ROI threshold [%]. Default: 15%.

        Returns:
            plotly.graph_objects.Figure with gauge chart.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        # Determine color based on ROI vs target
        if roi >= target_roi:
            color = colors.gauge_good_color
            status = 'Attractive'
        elif roi >= target_roi * 0.5:
            color = colors.gauge_medium_color
            status = 'Marginal'
        else:
            color = colors.gauge_bad_color
            status = 'Unattractive'

        # Create gauge steps with theme-aware colors
        from flex_model.visualization.core.color_schemes import get_rgba_with_alpha
        bad_color_alpha = get_rgba_with_alpha(colors.gauge_bad_color, 0.2)
        medium_color_alpha = get_rgba_with_alpha(colors.gauge_medium_color, 0.2)
        good_color_alpha = get_rgba_with_alpha(colors.gauge_good_color, 0.2)

        # Create gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=roi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"ROI<br><span style='font-size:0.8em;color:gray'>{status}</span>"},
            delta={'reference': target_roi, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, max(target_roi * 2, roi * 1.2)]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, target_roi * 0.5], 'color': bad_color_alpha},
                    {'range': [target_roi * 0.5, target_roi], 'color': medium_color_alpha},
                    {'range': [target_roi, target_roi * 2], 'color': good_color_alpha}
                ],
                'threshold': {
                    'line': {'color': colors.neutral_indicator_color, 'width': 4},
                    'thickness': 0.75,
                    'value': target_roi
                }
            },
            number={'suffix': '%', 'font': {'size': 50}},
        ))

        fig.update_layout(
            template=template,
            height=400,
        )

        return fig

    @staticmethod
    def create_payback_timeline(
        payback_period: float,
        lifetime_years: float = 10.0,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create payback period timeline visualization.

        Shows when investment breaks even within asset lifetime.

        Args:
            payback_period:
                Years to break even.

            lifetime_years:
                Expected asset lifetime [years].

            template:
                Plotly template name ('plotly_white', 'plotly_dark', 'plotly', etc.)

        Returns:
            plotly.graph_objects.Figure with timeline chart.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        # Create timeline
        fig = go.Figure()

        # Add lifetime bar
        fig.add_trace(go.Bar(
            x=[lifetime_years],
            y=['Investment'],
            orientation='h',
            marker_color=colors.neutral_indicator_color,
            name='Asset Lifetime',
            text=f'{lifetime_years:.0f} years',
            textposition='inside',
        ))

        # Add payback bar
        if payback_period <= lifetime_years:
            fig.add_trace(go.Bar(
                x=[payback_period],
                y=['Investment'],
                orientation='h',
                marker_color=colors.gauge_good_color,
                name='Payback Period',
                text=f'{payback_period:.1f} years',
                textposition='inside',
            ))
            status = '✓ Pays back within lifetime'
            status_color = colors.gauge_good_color
        else:
            fig.add_trace(go.Bar(
                x=[lifetime_years],
                y=['Investment'],
                orientation='h',
                marker_color=colors.gauge_bad_color,
                name='Does Not Pay Back',
                text=f'{payback_period:.1f} years (exceeds lifetime)',
                textposition='inside',
            ))
            status = '✗ Does not pay back'
            status_color = colors.gauge_bad_color

        fig.update_layout(
            title=f'Payback Period Analysis<br><sub style="color:{status_color}">{status}</sub>',
            xaxis_title='Years',
            template=template,
            showlegend=False,
            height=300,
        )

        return fig

    @staticmethod
    def create_financial_dashboard(
        metrics: Dict[str, Any],
        baseline_cost: float,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create comprehensive financial dashboard with multiple KPI panels.

        Combines multiple visualizations into a single executive summary.

        Args:
            metrics:
                Dictionary from EconomicMetrics.compute_financial_summary()

            baseline_cost:
                Annual baseline cost for reference [CHF/year]

        Returns:
            plotly.graph_objects.Figure with multi-panel dashboard.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")
        # Get color scheme for this template
        colors = get_color_scheme(template)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'ROI',
                'Payback Period',
                'Annual Cost Comparison',
                'Key Metrics'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'table'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        # Panel 1: ROI Gauge
        roi = metrics['roi']
        roi_color = colors.gauge_good_color if roi > 15 else colors.gauge_medium_color
        from flex_model.visualization.core.color_schemes import get_rgba_with_alpha
        bad_alpha = get_rgba_with_alpha(colors.gauge_bad_color, 0.2)
        good_alpha = get_rgba_with_alpha(colors.gauge_good_color, 0.2)

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=roi,
            title={'text': "ROI [%]"},
            gauge={
                'axis': {'range': [None, 50]},
                'bar': {'color': roi_color},
                'steps': [
                    {'range': [0, 15], 'color': bad_alpha},
                    {'range': [15, 50], 'color': good_alpha}
                ],
            },
            number={'suffix': '%'},
        ), row=1, col=1)

        # Panel 2: Payback Period
        payback = metrics['payback_period']
        lifetime = metrics['assumptions']['lifetime_years']
        payback_color = colors.gauge_good_color if payback <= lifetime else colors.gauge_bad_color

        fig.add_trace(go.Bar(
            x=[min(payback, lifetime)],
            y=['Payback'],
            orientation='h',
            marker_color=payback_color,
            text=[f'{payback:.1f} years'],
            textposition='inside',
        ), row=1, col=2)

        # Panel 3: Cost Comparison
        savings = metrics['savings']
        fig.add_trace(go.Bar(
            x=['Baseline', 'Optimized'],
            y=[savings['baseline_cost'], savings['optimized_cost']],
            marker_color=[colors.cost_color, colors.revenue_color],
            text=[f"{savings['baseline_cost']:,.0f}", f"{savings['optimized_cost']:,.0f}"],
            textposition='auto',
        ), row=2, col=1)

        # Panel 4: Key Metrics Table
        # Note: Table colors are intentionally simple/neutral for readability
        fig.add_trace(go.Table(
            header=dict(values=['Metric', 'Value'],
                       fill_color=colors.neutral_indicator_color,
                       align='left'),
            cells=dict(values=[
                ['NPV', 'LCOE', 'Annual Savings', 'Investment'],
                [f"{metrics['npv']:,.0f} CHF",
                 f"{metrics['lcoe']:.4f} CHF/kWh",
                 f"{savings['absolute_savings']:,.0f} CHF/year",
                 f"{savings['investment_required']:,.0f} CHF"]
            ],
            fill_color=get_rgba_with_alpha(colors.neutral_indicator_color, 0.3),
            align='left')
        ), row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text='Financial Summary Dashboard',
            showlegend=False,
            template=template,
            height=800,
        )

        return fig

    @staticmethod
    def create_investment_summary(
        investment_cost: float,
        capacity: float,
        unit_cost: float,
        lifetime_years: float,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create investment summary card with key numbers.

        Args:
            investment_cost:
                Total investment cost [CHF].

            capacity:
                Battery capacity [kWh].

            unit_cost:
                Cost per kWh [CHF/kWh].

            lifetime_years:
                Expected asset lifetime [years].

        Returns:
            plotly.graph_objects.Figure with investment summary.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Calculate annualized cost
        annual_cost = investment_cost / lifetime_years

        # Create figure with indicator
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="number",
            value=investment_cost,
            title={
                'text': f"<b>Investment Required</b><br>"
                       f"<span style='font-size:0.7em;color:gray'>{capacity:.0f} kWh @ {unit_cost:.0f} CHF/kWh</span><br>"
                       f"<span style='font-size:0.7em;color:gray'>Lifetime: {lifetime_years:.0f} years</span><br>"
                       f"<span style='font-size:0.7em;color:orange'>Annual: {annual_cost:,.0f} CHF/year</span>"
            },
            number={
                'prefix': "CHF ",
                'font': {'size': 50},
                'valueformat': ',.0f'
            },
            domain={'x': [0, 1], 'y': [0, 1]}
        ))

        fig.update_layout(
            template=template,
            height=350,
        )

        return fig

    @staticmethod
    def create_irr_gauge(
        irr: float,
        target_irr: float = 12.0,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create IRR gauge visualization.

        Visual indicator of internal rate of return with target benchmark.

        Args:
            irr:
                Internal rate of return [%].

            target_irr:
                Target IRR threshold [%]. Default: 12%.

        Returns:
            plotly.graph_objects.Figure with gauge chart.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        # Determine color based on IRR vs target
        if irr >= 15.0:
            color = colors.gauge_good_color
            status = 'Attractive'
        elif irr >= 10.0:
            color = colors.gauge_medium_color
            status = 'Competitive'
        else:
            color = colors.gauge_bad_color
            status = 'Below Market'

        # Create gauge steps with theme-aware colors
        from flex_model.visualization.core.color_schemes import get_rgba_with_alpha
        bad_color_alpha = get_rgba_with_alpha(colors.gauge_bad_color, 0.2)
        medium_color_alpha = get_rgba_with_alpha(colors.gauge_medium_color, 0.2)
        good_color_alpha = get_rgba_with_alpha(colors.gauge_good_color, 0.2)

        # Create gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=irr,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"IRR<br><span style='font-size:0.8em;color:gray'>{status}</span>"},
            delta={'reference': target_irr, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, max(target_irr * 2, irr * 1.2)]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 10], 'color': bad_color_alpha},
                    {'range': [10, 15], 'color': medium_color_alpha},
                    {'range': [15, target_irr * 2], 'color': good_color_alpha}
                ],
                'threshold': {
                    'line': {'color': colors.neutral_indicator_color, 'width': 4},
                    'thickness': 0.75,
                    'value': target_irr
                }
            },
            number={'suffix': '%', 'font': {'size': 50}},
        ))

        fig.update_layout(
            template=template,
            height=400,
        )

        return fig

    @staticmethod
    def create_cost_revenue_waterfall(
        breakdown: Dict[str, Any],
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create waterfall chart showing cost and revenue components.

        Args:
            breakdown:
                Dictionary from EconomicMetrics.compute_cost_revenue_breakdown()

            template:
                Plotly template name ('plotly_white', 'plotly_dark', 'plotly', etc.)

        Returns:
            plotly.graph_objects.Figure with waterfall chart.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        # Extract values
        costs = breakdown['costs']
        revenues = breakdown['revenues']

        # Build waterfall data
        labels = [
            'Start',
            'Battery\nDepreciation',
            'Battery\nO&M',
            'Market\nImport',
            'Market\nExport',
            'Avoided\nBaseline',
            'Net Cost'
        ]

        values = [
            0,  # Start at 0
            costs['battery_depreciation'],
            costs['battery_om'],
            costs['market_import_cost'],
            -revenues['market_export_revenue'],  # Negative because it's revenue
            -revenues['avoided_baseline'],        # Negative because it's savings
            breakdown['net']['net_annual_cost']
        ]

        measures = ['absolute', 'relative', 'relative', 'relative', 'relative', 'relative', 'total']

        # Determine connector color based on theme
        connector_color = 'rgb(100, 100, 100)' if 'dark' in template else 'rgb(63, 63, 63)'

        # Create waterfall
        fig = go.Figure(go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            text=[f"<b>{v:,.0f}</b>" for v in values],
            textposition='outside',
            textfont=dict(color=colors.annotation_color),
            connector={"line": {"color": connector_color}},
            increasing={"marker": {"color": colors.cost_color}},
            decreasing={"marker": {"color": colors.revenue_color}},
            totals={"marker": {"color": colors.total_color}}
        ))

        fig.update_layout(
            title='Annual Cost & Revenue Breakdown',
            yaxis_title='Amount [CHF/year]',
            template=template,
            showlegend=False,
            height=500,
        )

        return fig

    @staticmethod
    def create_daily_cost_timeseries(
        daily_profile: Dict[str, Any],
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create bar chart of daily costs over time.

        Args:
            daily_profile:
                Dictionary from EconomicMetrics.compute_daily_cost_profile()

            template:
                Plotly template name ('plotly_white', 'plotly_dark', 'plotly', etc.)

        Returns:
            plotly.graph_objects.Figure with daily cost timeline.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        daily_net = daily_profile['daily_net']
        days = list(range(len(daily_net)))
        stats = daily_profile['statistics']

        # Create figure
        fig = go.Figure()

        # Add daily cost bars (color by cost vs savings)
        fig.add_trace(go.Bar(
            x=days,
            y=daily_net,
            name='Daily Net Cost',
            marker_color=[colors.cost_color if v > stats['mean'] else colors.revenue_color for v in daily_net],
        ))

        # Add mean line
        fig.add_hline(
            y=stats['mean'],
            line_dash="dash",
            line_color=colors.neutral_indicator_color,
            annotation_text=f"Mean: {stats['mean']:.2f} CHF/day",
            annotation_position="right"
        )

        # Add ±1 std band
        from flex_model.visualization.core.color_schemes import get_rgba_with_alpha
        band_color = get_rgba_with_alpha(colors.neutral_indicator_color, 0.1)

        fig.add_hrect(
            y0=stats['mean'] - stats['std'],
            y1=stats['mean'] + stats['std'],
            fillcolor=band_color,
            line_width=0,
            annotation_text="±1 std dev",
            annotation_position="left"
        )

        fig.update_layout(
            title='Daily Cost Timeline',
            xaxis_title='Day',
            yaxis_title='Net Cost [CHF/day]',
            template=template,
            showlegend=False,
            height=400,
        )

        return fig

    @staticmethod
    def create_cost_variability_analysis(
        daily_profile: Dict[str, Any],
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create box plot showing daily cost distribution.

        Args:
            daily_profile:
                Dictionary from EconomicMetrics.compute_daily_cost_profile()

        Returns:
            plotly.graph_objects.Figure with box plot and statistics.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        daily_net = daily_profile['daily_net']
        stats = daily_profile['statistics']

        # Create box plot
        fig = go.Figure()

        # Use neutral colors for box plot
        from flex_model.visualization.core.color_schemes import get_rgba_with_alpha
        box_marker_color = get_rgba_with_alpha(colors.total_color, 0.7)
        box_fill_color = get_rgba_with_alpha(colors.total_color, 0.3)

        fig.add_trace(go.Box(
            y=daily_net,
            name='Daily Cost',
            boxmean='sd',  # Show mean and std dev
            marker_color=box_marker_color,
            fillcolor=box_fill_color,
        ))

        # Add statistics as annotations
        variability_pct = (stats['std'] / stats['mean'] * 100) if stats['mean'] != 0 else 0

        annotations_text = f"""
        <b>Statistics:</b><br>
        Mean: {stats['mean']:.2f} CHF/day<br>
        Median: {stats['median']:.2f} CHF/day<br>
        Std Dev: {stats['std']:.2f} CHF/day<br>
        Range: {stats['min']:.2f} to {stats['max']:.2f}<br>
        <br>
        <b>Variability: ±{variability_pct:.1f}%</b>
        """

        fig.add_annotation(
            x=0.5,
            y=stats['max'] * 1.1,
            text=annotations_text,
            showarrow=False,
            xref='x',
            yref='y',
            align='left',
            bgcolor=colors.annotation_color,
            bordercolor='gray',
            borderwidth=2,
        )

        fig.update_layout(
            title='Daily Cost Variability',
            yaxis_title='Net Cost [CHF/day]',
            template=template,
            showlegend=False,
            height=500,
        )

        return fig

    @staticmethod
    def create_investment_sensitivity_chart(
        sensitivity: Dict[str, Any],
        current_multiplier: float = 1.0,
        template: str = 'plotly_white',
    ) -> Any:  # go.Figure
        """
        Create multi-line chart showing how metrics change with investment cost.

        Args:
            sensitivity:
                Dictionary from EconomicMetrics.compute_investment_sensitivity()

            current_multiplier:
                Highlight current scenario (default 1.0)

        Returns:
            plotly.graph_objects.Figure with sensitivity analysis.

        Raises:
            ImportError: If plotly not installed.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualizations.")

        # Get color scheme for this template
        colors = get_color_scheme(template)

        multipliers = sensitivity['multipliers']
        roi_values = sensitivity['roi']
        irr_values = sensitivity['irr']
        payback_values = sensitivity['payback']
        breakeven = sensitivity['breakeven_multiplier']

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # ROI and IRR on primary axis
        fig.add_trace(
            go.Scatter(
                x=multipliers,
                y=roi_values,
                mode='lines+markers',
                name='ROI',
                line=dict(color=colors.total_color, width=2),
                marker=dict(size=8),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=multipliers,
                y=irr_values,
                mode='lines+markers',
                name='IRR',
                line=dict(color=colors.revenue_color, width=2),
                marker=dict(size=8),
            ),
            secondary_y=False,
        )

        # Payback on secondary axis
        # Cap payback at 2x lifetime for visualization
        max_display_payback = 20
        payback_display = [min(p, max_display_payback) for p in payback_values]

        fig.add_trace(
            go.Scatter(
                x=multipliers,
                y=payback_display,
                mode='lines+markers',
                name='Payback Period',
                line=dict(color=colors.gauge_medium_color, width=2, dash='dash'),
                marker=dict(size=8),
            ),
            secondary_y=True,
        )

        # Add vertical line for current scenario
        fig.add_vline(
            x=current_multiplier,
            line_dash="dot",
            line_color=colors.cost_color,
            annotation_text=f"Current ({current_multiplier:.1f}x)",
            annotation_position="top"
        )

        # Add vertical line for break-even
        if breakeven and breakeven != float('inf') and breakeven > 0:
            fig.add_vline(
                x=breakeven,
                line_dash="dot",
                line_color=colors.neutral_indicator_color,
                annotation_text=f"Break-even ({breakeven:.2f}x)",
                annotation_position="bottom"
            )

        # Update layout
        fig.update_xaxes(title_text="Investment Cost Multiplier")
        fig.update_yaxes(title_text="ROI / IRR [%]", secondary_y=False)
        fig.update_yaxes(title_text="Payback Period [years]", secondary_y=True)

        fig.update_layout(
            title='Investment Cost Sensitivity Analysis',
            template=template,
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig
