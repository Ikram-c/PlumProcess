import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import messagebox, filedialog
import webbrowser
import tempfile
import os
import subprocess
import shutil
from pathlib import Path


def load_coco_annotations(json_path):
    """Load COCO annotations from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def discover_annotation_properties(coco_data):
    """Discover all properties present in annotations"""
    all_properties = set()
    property_types = {}
    sample_values = defaultdict(set)

    # Analyze first 100 annotations to discover properties
    sample_size = min(100, len(coco_data['annotations']))

    for ann in coco_data['annotations'][:sample_size]:
        for key, value in ann.items():
            all_properties.add(key)
            sample_values[key].add(type(value).__name__)

            # Store sample values to determine data type
            if key not in property_types:
                if isinstance(value, (int, float)):
                    property_types[key] = 'numerical'
                elif isinstance(value, bool):
                    property_types[key] = 'boolean'
                elif isinstance(value, str):
                    property_types[key] = 'categorical'
                elif isinstance(value, list):
                    property_types[key] = 'list'
                else:
                    property_types[key] = 'other'

    return all_properties, property_types, sample_values


def extract_annotation_data(coco_data):
    """Extract all annotation data into a structured format"""
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Build comprehensive dataset
    annotation_data = []

    for ann in coco_data['annotations']:
        row = {}

        # Basic properties
        row['annotation_id'] = ann.get('id', None)
        row['image_id'] = ann.get('image_id', None)
        row['category_id'] = ann.get('category_id', None)
        row['category_name'] = categories.get(ann.get('category_id'), 'Unknown')

        # Bounding box properties
        if 'bbox' in ann:
            bbox = ann['bbox']
            row['bbox_x'] = bbox[0]
            row['bbox_y'] = bbox[1]
            row['bbox_width'] = bbox[2]
            row['bbox_height'] = bbox[3]
            row['bbox_area'] = bbox[2] * bbox[3]
            row['bbox_aspect_ratio'] = bbox[2] / bbox[3] if bbox[3] > 0 else 0

        # Standard COCO properties
        row['area'] = ann.get('area', None)
        row['iscrowd'] = ann.get('iscrowd', None)

        # Custom properties (exposure, color, shape, etc.)
        for key, value in ann.items():
            if key not in ['id', 'image_id', 'category_id', 'bbox', 'area', 'iscrowd', 'segmentation']:
                row[key] = value

        annotation_data.append(row)

    return pd.DataFrame(annotation_data)


def analyze_categorical_property(df, property_name, category_col='category_name'):
    """Analyze a categorical property across different categories"""
    # Count occurrences
    counts = df.groupby([category_col, property_name]).size().reset_index(name='count')

    # Calculate percentages within each category
    category_totals = df.groupby(category_col).size()
    counts['percentage'] = counts.apply(
        lambda row: (row['count'] / category_totals[row[category_col]]) * 100,
        axis=1
    )

    return counts


def analyze_numerical_property(df, property_name, category_col='category_name'):
    """Analyze a numerical property across different categories"""
    # Filter out non-null values
    clean_df = df[df[property_name].notna() & df[category_col].notna()]

    if clean_df.empty:
        return None

    # Calculate statistics by category
    stats = clean_df.groupby(category_col)[property_name].agg([
        'count', 'mean', 'std', 'min', 'max',
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 50),
        lambda x: np.percentile(x, 75)
    ]).round(3)

    stats.columns = ['count', 'mean', 'std', 'min', 'max', 'q25', 'median', 'q75']

    return stats, clean_df


def create_main_dashboard_with_class_selector(df, categories):
    """Create main dashboard with individual class selection dropdown"""

    # Create traces for each view type
    all_traces = []
    trace_info = []

    # 1. Overall category distribution (always visible)
    cat_counts = df['category_name'].value_counts()
    all_traces.append(go.Bar(
        x=cat_counts.index,
        y=cat_counts.values,
        name='All Categories',
        visible=True,
        marker_color='steelblue'
    ))
    trace_info.append({'type': 'overview', 'class': 'all'})

    # 2. Individual class detailed views
    for category in categories:
        cat_df = df[df['category_name'] == category]

        # Area distribution for this class
        if 'bbox_area' in cat_df.columns:
            areas = cat_df['bbox_area'].dropna()
            if not areas.empty:
                all_traces.append(go.Histogram(
                    x=areas,
                    nbinsx=30,
                    name=f'{category} - Area Distribution',
                    visible=False,
                    marker_color='lightcoral'
                ))
                trace_info.append({'type': 'area_hist', 'class': category})

        # Aspect ratio distribution for this class
        if 'bbox_aspect_ratio' in cat_df.columns:
            ratios = cat_df['bbox_aspect_ratio'].dropna()
            if not ratios.empty:
                all_traces.append(go.Histogram(
                    x=ratios,
                    nbinsx=30,
                    name=f'{category} - Aspect Ratio Distribution',
                    visible=False,
                    marker_color='lightgreen'
                ))
                trace_info.append({'type': 'ratio_hist', 'class': category})

        # Area vs Aspect Ratio scatter for this class
        if 'bbox_area' in cat_df.columns and 'bbox_aspect_ratio' in cat_df.columns:
            clean_cat_df = cat_df[cat_df['bbox_area'].notna() & cat_df['bbox_aspect_ratio'].notna()]
            if not clean_cat_df.empty:
                all_traces.append(go.Scatter(
                    x=clean_cat_df['bbox_aspect_ratio'],
                    y=clean_cat_df['bbox_area'],
                    mode='markers',
                    name=f'{category} - Area vs Aspect Ratio',
                    visible=False,
                    marker=dict(color='orange', opacity=0.7)
                ))
                trace_info.append({'type': 'scatter', 'class': category})

        # Annotations per image for this class
        img_counts = cat_df.groupby('image_id').size()
        if not img_counts.empty:
            all_traces.append(go.Histogram(
                x=img_counts.values,
                nbinsx=20,
                name=f'{category} - Annotations per Image',
                visible=False,
                marker_color='mediumpurple'
            ))
            trace_info.append({'type': 'img_hist', 'class': category})

    # Create dropdown menu options
    dropdown_buttons = []

    # Overall view
    visible_overall = [trace['type'] == 'overview' for trace in trace_info]
    dropdown_buttons.append(dict(
        label='📊 All Categories Overview',
        method='update',
        args=[{'visible': visible_overall},
              {'title': 'All Categories Overview',
               'xaxis': {'title': 'Category'},
               'yaxis': {'title': 'Count'}}]
    ))

    # Individual class views
    for category in categories:
        # Area distribution view
        visible_area = [trace['type'] == 'area_hist' and trace['class'] == category for trace in trace_info]
        if any(visible_area):
            dropdown_buttons.append(dict(
                label=f'📏 {category} - Area Distribution',
                method='update',
                args=[{'visible': visible_area},
                      {'title': f'{category} - Bounding Box Area Distribution',
                       'xaxis': {'title': 'Area (pixels²)'},
                       'yaxis': {'title': 'Count'}}]
            ))

        # Aspect ratio distribution view
        visible_ratio = [trace['type'] == 'ratio_hist' and trace['class'] == category for trace in trace_info]
        if any(visible_ratio):
            dropdown_buttons.append(dict(
                label=f'📐 {category} - Aspect Ratio Distribution',
                method='update',
                args=[{'visible': visible_ratio},
                      {'title': f'{category} - Aspect Ratio Distribution',
                       'xaxis': {'title': 'Aspect Ratio (width/height)'},
                       'yaxis': {'title': 'Count'}}]
            ))

        # Scatter plot view
        visible_scatter = [trace['type'] == 'scatter' and trace['class'] == category for trace in trace_info]
        if any(visible_scatter):
            dropdown_buttons.append(dict(
                label=f'🔍 {category} - Area vs Aspect Ratio',
                method='update',
                args=[{'visible': visible_scatter},
                      {'title': f'{category} - Area vs Aspect Ratio',
                       'xaxis': {'title': 'Aspect Ratio (width/height)'},
                       'yaxis': {'title': 'Area (pixels²)'}}]
            ))

        # Annotations per image view
        visible_img = [trace['type'] == 'img_hist' and trace['class'] == category for trace in trace_info]
        if any(visible_img):
            dropdown_buttons.append(dict(
                label=f'🖼️ {category} - Annotations per Image',
                method='update',
                args=[{'visible': visible_img},
                      {'title': f'{category} - Annotations per Image',
                       'xaxis': {'title': 'Number of Annotations'},
                       'yaxis': {'title': 'Number of Images'}}]
            ))

    # Create figure with dropdown
    fig = go.Figure(data=all_traces)

    fig.update_layout(
        title='COCO Annotations Dashboard - Select Class and View Type',
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
        ],
        height=600,
        xaxis_title='Category',
        yaxis_title='Count'
    )

    return fig
    """Analyze a categorical property across different categories"""
    # Count occurrences
    counts = df.groupby([category_col, property_name]).size().reset_index(name='count')

    # Calculate percentages within each category
    category_totals = df.groupby(category_col).size()
    counts['percentage'] = counts.apply(
        lambda row: (row['count'] / category_totals[row[category_col]]) * 100,
        axis=1
    )

    return counts


def analyze_numerical_property(df, property_name, category_col='category_name'):
    """Analyze a numerical property across different categories"""
    # Filter out non-null values
    clean_df = df[df[property_name].notna() & df[category_col].notna()]

    if clean_df.empty:
        return None

    # Calculate statistics by category
    stats = clean_df.groupby(category_col)[property_name].agg([
        'count', 'mean', 'std', 'min', 'max',
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 50),
        lambda x: np.percentile(x, 75)
    ]).round(3)

    stats.columns = ['count', 'mean', 'std', 'min', 'max', 'q25', 'median', 'q75']

    return stats, clean_df


def create_categorical_analysis_with_class_selector(df, prop, categories):
    """Create categorical property analysis with class selector dropdown"""

    # Create traces for each view
    all_traces = []
    trace_info = []

    # 1. Overall distribution across all classes
    overall_counts = analyze_categorical_property(df, prop)

    # Stacked bar chart for all classes
    prop_values = sorted(df[prop].dropna().unique())
    for prop_val in prop_values:
        val_data = overall_counts[overall_counts[prop] == prop_val]
        all_traces.append(go.Bar(
            name=f'{prop_val} (All)',
            x=val_data['category_name'],
            y=val_data['count'],
            text=val_data['percentage'].round(1).astype(str) + '%',
            textposition='inside',
            visible=True
        ))
        trace_info.append({'type': 'overall', 'prop_val': prop_val})

    # 2. Individual class analyses
    for category in categories:
        cat_df = df[df['category_name'] == category]
        if cat_df.empty or cat_df[prop].isna().all():
            continue

        # Value distribution for this specific class
        value_counts = cat_df[prop].value_counts()
        total_cat = len(cat_df)
        percentages = (value_counts / total_cat * 100).round(1)

        all_traces.append(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            text=[f'{count}<br>({pct}%)' for count, pct in zip(value_counts.values, percentages.values)],
            textposition='inside',
            name=f'{category} Distribution',
            visible=False,
            marker_color='lightcoral'
        ))
        trace_info.append({'type': 'individual', 'class': category})

    # Create dropdown buttons
    dropdown_buttons = []

    # Overall view button
    visible_overall = [trace['type'] == 'overall' for trace in trace_info]
    dropdown_buttons.append(dict(
        label='📊 All Classes Combined',
        method='update',
        args=[{'visible': visible_overall},
              {'title': f'{prop.replace("_", " ").title()} Distribution - All Classes',
               'xaxis': {'title': 'Category'},
               'yaxis': {'title': 'Count'},
               'barmode': 'stack'}]
    ))

    # Individual class buttons
    for category in categories:
        visible_individual = [trace['type'] == 'individual' and trace['class'] == category for trace in trace_info]
        if any(visible_individual):
            dropdown_buttons.append(dict(
                label=f'🔍 {category} Only',
                method='update',
                args=[{'visible': visible_individual},
                      {'title': f'{prop.replace("_", " ").title()} Distribution - {category}',
                       'xaxis': {'title': f'{prop.replace("_", " ").title()} Value'},
                       'yaxis': {'title': 'Count'},
                       'barmode': 'group'}]
            ))

    # Create figure
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f'{prop.replace("_", " ").title()} Analysis by Class',
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
        ],
        height=600,
        xaxis_title='Category',
        yaxis_title='Count',
        barmode='stack'
    )

    return fig


def create_numerical_analysis_with_class_selector(df, prop, categories):
    """Create numerical property analysis with class selector dropdown"""

    # Filter valid data
    clean_df = df[df[prop].notna() & df['category_name'].notna()]
    if clean_df.empty:
        return None

    # Create traces for each view
    all_traces = []
    trace_info = []

    # 1. Box plot comparing all classes
    for cat in categories:
        cat_data = clean_df[clean_df['category_name'] == cat][prop]
        if not cat_data.empty:
            all_traces.append(go.Box(
                y=cat_data,
                name=cat,
                boxmean=True,
                visible=True
            ))
            trace_info.append({'type': 'boxplot', 'class': cat})

    # 2. Overall histogram
    all_traces.append(go.Histogram(
        x=clean_df[prop],
        nbinsx=30,
        name='All Classes',
        visible=False,
        marker_color='steelblue'
    ))
    trace_info.append({'type': 'overall_hist', 'class': 'all'})

    # 3. Individual class histograms
    for category in categories:
        cat_data = clean_df[clean_df['category_name'] == category][prop]
        if not cat_data.empty:
            all_traces.append(go.Histogram(
                x=cat_data,
                nbinsx=20,
                name=f'{category} Distribution',
                visible=False,
                opacity=0.7
            ))
            trace_info.append({'type': 'individual_hist', 'class': category})

    # 4. Individual class statistics (as scatter plots for visualization)
    stats_df = clean_df.groupby('category_name')[prop].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)

    all_traces.append(go.Scatter(
        x=stats_df.index,
        y=stats_df['mean'],
        error_y=dict(type='data', array=stats_df['std']),
        mode='markers+lines',
        name='Mean ± Std',
        visible=False,
        marker=dict(size=10)
    ))
    trace_info.append({'type': 'stats', 'class': 'all'})

    # Create dropdown buttons
    dropdown_buttons = []

    # Box plot comparison
    visible_boxplot = [trace['type'] == 'boxplot' for trace in trace_info]
    dropdown_buttons.append(dict(
        label='📊 Box Plot Comparison (All Classes)',
        method='update',
        args=[{'visible': visible_boxplot},
              {'title': f'{prop.replace("_", " ").title()} Distribution by Class',
               'xaxis': {'title': 'Category'},
               'yaxis': {'title': prop.replace("_", " ").title()}}]
    ))

    # Overall histogram
    visible_overall_hist = [trace['type'] == 'overall_hist' for trace in trace_info]
    dropdown_buttons.append(dict(
        label='📈 Overall Distribution (All Classes)',
        method='update',
        args=[{'visible': visible_overall_hist},
              {'title': f'{prop.replace("_", " ").title()} Overall Distribution',
               'xaxis': {'title': prop.replace("_", " ").title()},
               'yaxis': {'title': 'Count'}}]
    ))

    # Statistics view
    visible_stats = [trace['type'] == 'stats' for trace in trace_info]
    dropdown_buttons.append(dict(
        label='📋 Statistical Summary',
        method='update',
        args=[{'visible': visible_stats},
              {'title': f'{prop.replace("_", " ").title()} Statistics by Class',
               'xaxis': {'title': 'Category'},
               'yaxis': {'title': f'Mean {prop.replace("_", " ").title()}'}}]
    ))

    # Individual class histograms
    for category in categories:
        visible_individual = [trace['type'] == 'individual_hist' and trace['class'] == category for trace in trace_info]
        if any(visible_individual):
            dropdown_buttons.append(dict(
                label=f'🔍 {category} Distribution',
                method='update',
                args=[{'visible': visible_individual},
                      {'title': f'{prop.replace("_", " ").title()} Distribution - {category}',
                       'xaxis': {'title': prop.replace("_", " ").title()},
                       'yaxis': {'title': 'Count'}}]
            ))

    # Create figure
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f'{prop.replace("_", " ").title()} Analysis by Class',
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
        ],
        height=600,
        xaxis_title='Category',
        yaxis_title=prop.replace("_", " ").title()
    )

    return fig


def create_boolean_analysis_with_class_selector(df, prop, categories):
    """Create boolean property analysis with class selector dropdown"""

    # Create traces for each view
    all_traces = []
    trace_info = []

    # 1. Overall comparison across all classes
    bool_counts = df.groupby(['category_name', prop]).size().reset_index(name='count')

    for bool_val in [True, False]:
        val_data = bool_counts[bool_counts[prop] == bool_val]
        all_traces.append(go.Bar(
            name=f'{bool_val} (All)',
            x=val_data['category_name'],
            y=val_data['count'],
            visible=True
        ))
        trace_info.append({'type': 'overall', 'bool_val': bool_val})

    # 2. Individual class analyses
    for category in categories:
        cat_df = df[df['category_name'] == category]
        if cat_df.empty or cat_df[prop].isna().all():
            continue

        # True/False counts for this class
        value_counts = cat_df[prop].value_counts()
        total_cat = len(cat_df[cat_df[prop].notna()])
        percentages = (value_counts / total_cat * 100).round(1)

        all_traces.append(go.Bar(
            x=[f'{val} ({pct}%)' for val, pct in zip(value_counts.index, percentages.values)],
            y=value_counts.values,
            name=f'{category}',
            visible=False,
            text=value_counts.values,
            textposition='inside'
        ))
        trace_info.append({'type': 'individual', 'class': category})

    # Create dropdown buttons
    dropdown_buttons = []

    # Overall comparison
    visible_overall = [trace['type'] == 'overall' for trace in trace_info]
    dropdown_buttons.append(dict(
        label='📊 All Classes Comparison',
        method='update',
        args=[{'visible': visible_overall},
              {'title': f'{prop.replace("_", " ").title()} Distribution - All Classes',
               'xaxis': {'title': 'Category'},
               'yaxis': {'title': 'Count'},
               'barmode': 'group'}]
    ))

    # Individual class views
    for category in categories:
        visible_individual = [trace['type'] == 'individual' and trace['class'] == category for trace in trace_info]
        if any(visible_individual):
            dropdown_buttons.append(dict(
                label=f'🔍 {category} Only',
                method='update',
                args=[{'visible': visible_individual},
                      {'title': f'{prop.replace("_", " ").title()} Distribution - {category}',
                       'xaxis': {'title': f'{prop.replace("_", " ").title()} Value'},
                       'yaxis': {'title': 'Count'},
                       'barmode': 'group'}]
            ))

    # Create figure
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f'{prop.replace("_", " ").title()} Analysis by Class',
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
        ],
        height=600,
        xaxis_title='Category',
        yaxis_title='Count',
        barmode='group'
    )


def create_summary_report(df, output_dir):
    """Create a comprehensive summary report"""
    # Discover properties
    all_properties, property_types, sample_values = discover_annotation_properties(
        {'annotations': df.to_dict('records')})

    # Create summary statistics
    summary = {
        'Basic Statistics': {
            'Total Annotations': len(df),
            'Unique Categories': df['category_name'].nunique(),
            'Unique Images': df['image_id'].nunique(),
            'Average Annotations per Image': len(df) / df['image_id'].nunique(),
        },
        'Discovered Properties': {
            'Total Properties': len(all_properties),
            'Numerical Properties': sum(1 for p, t in property_types.items() if t == 'numerical'),
            'Categorical Properties': sum(1 for p, t in property_types.items() if t == 'categorical'),
            'Boolean Properties': sum(1 for p, t in property_types.items() if t == 'boolean'),
        }
    }

    # Property details
    property_details = []
    for prop in sorted(all_properties):
        if prop in df.columns:
            non_null_count = df[prop].notna().sum()
            null_count = df[prop].isna().sum()
            prop_type = property_types.get(prop, 'unknown')

            detail = {
                'Property': prop,
                'Type': prop_type,
                'Non-null Count': non_null_count,
                'Null Count': null_count,
                'Fill Rate %': round((non_null_count / len(df)) * 100, 1)
            }

            if prop_type == 'numerical' and non_null_count > 0:
                detail['Min'] = df[prop].min()
                detail['Max'] = df[prop].max()
                detail['Mean'] = round(df[prop].mean(), 3)
                detail['Std'] = round(df[prop].std(), 3)
            elif prop_type == 'categorical' and non_null_count > 0:
                detail['Unique Values'] = df[prop].nunique()
                detail['Most Common'] = df[prop].mode().iloc[0] if not df[prop].mode().empty else 'N/A'

            property_details.append(detail)

    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>COCO Annotation Analysis Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .summary-box {{ background: #f9f9f9; padding: 20px; border-radius: 8px; }}
            .property-type {{ padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }}
            .type-numerical {{ background-color: #e3f2fd; color: #1976d2; }}
            .type-categorical {{ background-color: #f3e5f5; color: #7b1fa2; }}
            .type-boolean {{ background-color: #e8f5e9; color: #388e3c; }}
            .type-other {{ background-color: #fff3e0; color: #f57c00; }}
        </style>
    </head>
    <body>
        <h1>COCO Annotation Analysis Summary</h1>

        <div class="summary-grid">
            <div class="summary-box">
                <h3>Basic Statistics</h3>
                <ul>
    """

    for key, value in summary['Basic Statistics'].items():
        html_content += f"<li><strong>{key}:</strong> {value:,}</li>"

    html_content += """
                </ul>
            </div>
            <div class="summary-box">
                <h3>Discovered Properties</h3>
                <ul>
    """

    for key, value in summary['Discovered Properties'].items():
        html_content += f"<li><strong>{key}:</strong> {value}</li>"

    html_content += """
                </ul>
            </div>
        </div>

        <h2>Property Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Property</th>
                    <th>Type</th>
                    <th>Fill Rate %</th>
                    <th>Non-null Count</th>
                    <th>Additional Info</th>
                </tr>
            </thead>
            <tbody>
    """

    for detail in property_details:
        prop_type = detail['Type']
        type_class = f"type-{prop_type}"

        additional_info = ""
        if prop_type == 'numerical':
            additional_info = f"Range: {detail.get('Min', 'N/A')} - {detail.get('Max', 'N/A')}, Mean: {detail.get('Mean', 'N/A')}"
        elif prop_type == 'categorical':
            additional_info = f"Unique: {detail.get('Unique Values', 'N/A')}, Most Common: {detail.get('Most Common', 'N/A')}"

        html_content += f"""
                <tr>
                    <td><strong>{detail['Property']}</strong></td>
                    <td><span class="property-type {type_class}">{prop_type.title()}</span></td>
                    <td>{detail['Fill Rate %']}%</td>
                    <td>{detail['Non-null Count']:,}</td>
                    <td>{additional_info}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>

        <h2>Category Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Avg Area</th>
                    <th>Avg Aspect Ratio</th>
                </tr>
            </thead>
            <tbody>
    """

    # Category breakdown
    for category in sorted(df['category_name'].unique()):
        cat_data = df[df['category_name'] == category]
        count = len(cat_data)
        percentage = (count / len(df)) * 100

        # Handle avg_area formatting
        if 'bbox_area' in cat_data.columns:
            avg_area = cat_data['bbox_area'].mean()
            avg_area_str = f"{avg_area:.0f}" if not pd.isna(avg_area) else "N/A"
        else:
            avg_area_str = "N/A"

        # Handle avg_ratio formatting
        if 'bbox_aspect_ratio' in cat_data.columns:
            avg_ratio = cat_data['bbox_aspect_ratio'].mean()
            avg_ratio_str = f"{avg_ratio:.2f}" if not pd.isna(avg_ratio) else "N/A"
        else:
            avg_ratio_str = "N/A"

        html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{count:,}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{avg_area_str}</td>
                    <td>{avg_ratio_str}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Save summary report
    summary_path = os.path.join(output_dir, "summary_report.html")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return summary_path


def create_navigation_page(output_dir, main_dashboard, property_files, summary_report):
    """Create a navigation page linking to all analyses"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>COCO Analysis Navigation</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                background-color: #f5f5f5; 
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            h1 { 
                color: #333; 
                text-align: center; 
                margin-bottom: 30px; 
            }
            .nav-section { 
                margin: 30px 0; 
                padding: 20px; 
                background-color: #f9f9f9; 
                border-radius: 8px; 
            }
            .nav-section h2 { 
                color: #555; 
                margin-bottom: 15px; 
            }
            .nav-links { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 15px; 
            }
            .nav-link { 
                display: block; 
                padding: 15px; 
                background-color: #007bff; 
                color: white; 
                text-decoration: none; 
                border-radius: 6px; 
                transition: background-color 0.3s; 
                text-align: center; 
            }
            .nav-link:hover { 
                background-color: #0056b3; 
            }
            .summary-link { 
                background-color: #28a745; 
            }
            .summary-link:hover { 
                background-color: #1e7e34; 
            }
            .main-link { 
                background-color: #dc3545; 
            }
            .main-link:hover { 
                background-color: #c82333; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 COCO Annotation Analysis Dashboard</h1>

            <div class="nav-section">
                <h2>📊 Main Reports</h2>
                <div class="nav-links">
                    <a href="summary_report.html" class="nav-link summary-link">
                        📋 Summary Report
                    </a>
                    <a href="main_dashboard.html" class="nav-link main-link">
                        📈 Main Dashboard
                    </a>
                </div>
            </div>

            <div class="nav-section">
                <h2>🔍 Property-Specific Analyses</h2>
                <div class="nav-links">
    """

    # Add links to property analyses
    for prop, filename in sorted(property_files):
        display_name = prop.replace('_', ' ').title()
        html_content += f"""
                    <a href="{filename}" class="nav-link">
                        🏷️ {display_name}
                    </a>
        """

    html_content += """
                </div>
            </div>

            <div class="nav-section">
                <h2>ℹ️ Analysis Information</h2>
                <p>This analysis was generated from your COCO annotation data and includes:</p>
                <ul>
                    <li><strong>Summary Report:</strong> Overview of all discovered properties and basic statistics</li>
                    <li><strong>Main Dashboard:</strong> Interactive overview of categories, areas, and distributions</li>
                    <li><strong>Property Analyses:</strong> Detailed visualizations for each discovered property</li>
                </ul>
                <p>Each visualization is interactive - hover, zoom, and click to explore your data!</p>
            </div>
        </div>
    </body>
    </html>
    """

    nav_path = os.path.join(output_dir, "index.html")
    with open(nav_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return nav_path


def choose_json_file():
    """Open file dialog to choose COCO JSON file"""
    root = tk.Tk()
    root.withdraw()

    json_path = filedialog.askopenfilename(
        title="Select COCO JSON file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )

    root.destroy()
    return json_path


def choose_output_directory():
    """Open dialog to choose output directory"""
    root = tk.Tk()
    root.withdraw()

    output_dir = filedialog.askdirectory(
        title="Choose output directory for analysis"
    )

    root.destroy()
    return output_dir


def run_comprehensive_analysis():
    """Run the complete analysis pipeline"""
    try:
        # Choose input file
        json_path = choose_json_file()
        if not json_path:
            messagebox.showinfo("Info", "No JSON file selected")
            return

        # Choose output directory
        output_dir = choose_output_directory()
        if not output_dir:
            # Default to user's home directory
            output_dir = os.path.join(os.path.expanduser("~"), "coco_analysis")

        # Create timestamped subdirectory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"coco_analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Load and process data
        print(f"Loading COCO annotations from: {json_path}")
        coco_data = load_coco_annotations(json_path)

        print("Extracting annotation data...")
        df = extract_annotation_data(coco_data)

        print(f"Total annotations: {len(df)}")
        print(f"Unique categories: {df['category_name'].nunique()}")
        print(f"Unique images: {df['image_id'].nunique()}")

        # Discover properties
        all_properties, property_types, sample_values = discover_annotation_properties(coco_data)
        print(f"Discovered {len(all_properties)} properties:")
        for prop, prop_type in sorted(property_types.items()):
            print(f"  - {prop}: {prop_type}")

        # Create visualizations
        print("Creating visualizations...")
        main_dashboard, property_files = create_property_visualizations(df, output_dir)

        print("Creating summary report...")
        summary_report = create_summary_report(df, output_dir)

        print("Creating navigation page...")
        nav_page = create_navigation_page(output_dir, main_dashboard, property_files, summary_report)

        # Open navigation page
        try:
            webbrowser.open('file://' + os.path.abspath(nav_page))
            print(f"✓ Opened analysis in browser")
        except Exception as e:
            print(f"Could not open browser: {e}")

        messagebox.showinfo(
            "Analysis Complete!",
            f"Comprehensive analysis completed!\n\n"
            f"Results saved to: {output_dir}\n"
            f"Open index.html to view all analyses\n\n"
            f"Total properties analyzed: {len(property_files)}"
        )

        return output_dir

    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        print(error_msg)
        messagebox.showerror("Error", error_msg)
        return None


# GUI Application
def main():
    root = tk.Tk()
    root.title("Advanced COCO Annotation Analysis")
    root.geometry("500x400")

    # Title
    title_label = tk.Label(root, text="🔍 Advanced COCO Analysis", font=('Arial', 18, 'bold'))
    title_label.pack(pady=20)

    # Description
    desc_label = tk.Label(
        root,
        text="Comprehensive analysis of COCO annotations including:\n"
             "• All standard properties (bbox, area, etc.)\n"
             "• Custom properties (exposure, color, shape, etc.)\n"
             "• Interactive visualizations\n"
             "• Statistical summaries",
        font=('Arial', 11),
        justify='center'
    )
    desc_label.pack(pady=10)

    # Main button
    analyze_btn = tk.Button(
        root,
        text="🚀 Start Comprehensive Analysis",
        command=run_comprehensive_analysis,
        font=('Arial', 14, 'bold'),
        padx=30,
        pady=15,
        bg='#007bff',
        fg='white',
        cursor='hand2'
    )
    analyze_btn.pack(pady=30)

    # Instructions
    instructions = tk.Label(
        root,
        text="Instructions:\n"
             "1. Click 'Start Comprehensive Analysis'\n"
             "2. Select your COCO JSON file\n"
             "3. Choose output directory (optional)\n"
             "4. Wait for analysis to complete\n"
             "5. Explore results in your browser!",
        font=('Arial', 10),
        justify='left',
        wraplength=450
    )
    instructions.pack(pady=20)

    # Footer
    footer = tk.Label(root, text="Supports all COCO properties + custom extensions",
                      font=('Arial', 9), fg='gray')
    footer.pack(side='bottom', pady=10)

    root.mainloop()


if __name__ == '__main__':
    main()