#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import panel as pn
from skbio.stats.ordination import pcoa
from sklearn.manifold import MDS

# Enable Panel extensions
pn.extension()

# ============================================================
# File paths
# ============================================================
OUTPUT_FOLDER = "./saved_matrices"
MATRIX_FILE = os.path.join(OUTPUT_FOLDER, "braycurtis_matrix_columns.npy")
FEATURE_NAMES_FILE = os.path.join(OUTPUT_FOLDER, "feature_names.txt")
METADATA_FILE = "./metadata.csv"

# Intermediate product files
PCOA_RESULTS_FILE = os.path.join(OUTPUT_FOLDER, "pcoa_results.csv")
MERGED_DF_FILE = os.path.join(OUTPUT_FOLDER, "merged_df.csv")

# ============================================================
# 1. Load Saved Bray–Curtis Matrix and Sample/Feature Names
# ============================================================
if not os.path.exists(MATRIX_FILE) or not os.path.exists(FEATURE_NAMES_FILE):
    sys.exit("Error: Required BC matrix or feature names file not found in 'saved_matrices'.")

distance_matrix = np.load(MATRIX_FILE)
with open(FEATURE_NAMES_FILE, "r") as f:
    samples = [line.strip() for line in f if line.strip()]

# ============================================================
# 2. Compute or Load PCoA (or fallback to MDS)
# ============================================================
if os.path.exists(PCOA_RESULTS_FILE):
    print("Loading saved PCoA results...")
    pcoa_df = pd.read_csv(PCOA_RESULTS_FILE, index_col=0)
    used_pcoa = True
    # When loading saved results, we don't have actual proportions.
    prop_explained = np.array([0, 0])
else:
    used_pcoa = True
    try:
        pcoa_results = pcoa(distance_matrix)
        pcoa_df = pcoa_results.samples.iloc[:, :2].copy()
        pcoa_df.index = samples
        pcoa_df.columns = ["PC1", "PC2"]
        prop_explained = pcoa_results.proportion_explained.iloc[:2].values
    except Exception as e:
        print(f"PCoA failed due to: {e}")
        used_pcoa = False
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        coords = mds.fit_transform(distance_matrix)
        pcoa_df = pd.DataFrame(coords, index=samples, columns=["PC1", "PC2"])
        prop_explained = np.array([0, 0])
    pcoa_df.to_csv(PCOA_RESULTS_FILE)
    print("PCoA results computed and saved.")

print("PCoA/MDS coordinates (first 5 rows):")
print(pcoa_df.head())

# ============================================================
# 3. Load Metadata and Merge with PCoA Results
# ============================================================
try:
    metadata = pd.read_csv(METADATA_FILE, sep=",", index_col="#NAME")
    print("Metadata loaded. Index (first 5):", metadata.index[:5].tolist())
except FileNotFoundError:
    sys.exit(f"Error: Metadata file not found: {METADATA_FILE}")

if os.path.exists(MERGED_DF_FILE):
    print("Loading saved merged dataframe...")
    merged_df = pd.read_csv(MERGED_DF_FILE, index_col=0)
else:
    merged_df = pcoa_df.join(metadata, how="inner")
    if merged_df.empty:
        print("WARNING: The merged dataframe is empty. Check sample IDs vs 'Run' in metadata.")
    merged_df.to_csv(MERGED_DF_FILE)
    print("Merged dataframe computed and saved.")

# ============================================================
# 4. Precompute Zoom Bounds for Plotting
# ============================================================
x_min, x_max = merged_df["PC1"].min(), merged_df["PC1"].max()
y_min, y_max = merged_df["PC2"].min(), merged_df["PC2"].max()
x_center = 0.5 * (x_min + x_max)
y_center = 0.5 * (y_min + y_max)
x_half_range = 0.5 * (x_max - x_min)
y_half_range = 0.5 * (y_max - y_min)

# ============================================================
# 5. Define Plotting Function for Panel Interactivity
# ============================================================
def plot_pcoa(color_factor, shape_factor, show_names, marker_size, zoom, highlight_factor, highlight_cats, point_color=None, legend_marker_size=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set default legend marker size if not provided
    if legend_marker_size is None:
        legend_marker_size = marker_size * 1.5

    # Determine whether to use highlight mode
    use_highlight = (highlight_factor is not None) and (highlight_factor in merged_df.columns) \
                    and (highlight_cats is not None) and (len(highlight_cats) > 0)
    
    if use_highlight:
        others_df = merged_df[~merged_df[highlight_factor].isin(highlight_cats)]
        highlight_df = merged_df[merged_df[highlight_factor].isin(highlight_cats)]
        style_var = shape_factor if (shape_factor in merged_df.columns) else None
        sns.scatterplot(data=others_df, x="PC1", y="PC2", color="lightgray", style=style_var,
                        s=marker_size, edgecolor="black", zorder=1, ax=ax)
        palette = sns.color_palette("tab10", len(highlight_cats))
        for i, cat in enumerate(highlight_cats):
            sub = highlight_df[highlight_df[highlight_factor] == cat]
            sns.scatterplot(data=sub, x="PC1", y="PC2", color=palette[i], style=style_var,
                            s=marker_size, edgecolor="black", zorder=2, ax=ax, label=f"{highlight_factor}={cat}")
        if show_names:
            for sample, row in highlight_df.iterrows():
                x_val, y_val = row["PC1"], row["PC2"]
                if np.isfinite(x_val) and np.isfinite(y_val):
                    ax.text(x_val, y_val, sample, fontsize=9, ha="right")
        ax.set_title(f"PCoA - Highlight: '{highlight_factor}' -> {highlight_cats}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        # Use point_color if provided and no valid hue column is set.
        if color_factor is None or color_factor == "(None)" or color_factor not in merged_df.columns:
            scatter = sns.scatterplot(data=merged_df, x="PC1", y="PC2",
                                      color=point_color if point_color is not None else "blue",
                                      style=shape_factor if shape_factor != "(None)" and shape_factor in merged_df.columns else None,
                                      s=marker_size, edgecolor="black",
                                      ax=ax)
        else:
            custom_palette = sns.color_palette("tab20", 20)
            scatter = sns.scatterplot(data=merged_df, x="PC1", y="PC2",
                                      hue=color_factor, style=shape_factor if shape_factor != "(None)" and shape_factor in merged_df.columns else None,
                                      s=marker_size, edgecolor="black",
                                      palette=custom_palette, ax=ax)
        if show_names:
            for sample, row in merged_df.iterrows():
                x_val, y_val = row["PC1"], row["PC2"]
                if np.isfinite(x_val) and np.isfinite(y_val):
                    ax.text(x_val, y_val, sample, fontsize=9, ha="right")
        if used_pcoa and not np.allclose(prop_explained, [0, 0]):
            xlabel = f"PC1 ({prop_explained[0]*100:.2f}%)"
            ylabel = f"PC2 ({prop_explained[1]*100:.2f}%)"
        else:
            xlabel, ylabel = "PC1", "PC2"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"PCoA Plot - Color={color_factor}, Shape={shape_factor}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Increase legend marker sizes if legend exists (use legend_handles attribute)
    leg = ax.get_legend()
    if leg:
        for handle in leg.legend_handles:
            try:
                handle.set_sizes([legend_marker_size])
            except Exception:
                pass

    cur_x_half = x_half_range / zoom
    cur_y_half = y_half_range / zoom
    ax.set_xlim(x_center - cur_x_half, x_center + cur_x_half)
    ax.set_ylim(y_center - cur_y_half, y_center + cur_y_half)
    ax.grid(True)
    plt.close(fig)
    return fig

# ============================================================
# 6b. Function to Save Plot as PNG
# ============================================================
def save_plot_as_png(filename, color_factor, shape_factor, show_names, marker_size, zoom, highlight_factor, highlight_cats, point_color=None, legend_marker_size=None):
    """
    Generate a PCoA/MDS plot using the provided parameters and save it as a PNG file.
    """
    fig = plot_pcoa(color_factor, shape_factor, show_names, marker_size, zoom, highlight_factor, highlight_cats, point_color, legend_marker_size)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as PNG: {filename}")

# ============================================================
# 7. Create Panel Widgets
# ============================================================
color_dropdown = pn.widgets.Select(name="Select Color Factor", options=["(None)"] + list(metadata.columns), value="(None)")
shape_dropdown = pn.widgets.Select(name="Select Shape Factor", options=["(None)"] + list(metadata.columns), value="(None)")
show_names_checkbox = pn.widgets.Checkbox(name="Show Sample Names", value=True)
marker_size_slider = pn.widgets.IntSlider(name="Marker Size", value=100, start=10, end=300, step=10)
zoom_slider = pn.widgets.FloatSlider(name="Zoom", value=1.0, start=0.2, end=3.0, step=0.1)
legend_marker_size_slider = pn.widgets.IntSlider(name="Legend Marker Size", value=150, start=10, end=500, step=10)
highlight_factor_dropdown = pn.widgets.Select(name="Highlight Factor", options=["(None)"] + list(metadata.columns), value="(None)")
highlight_categories_widget = pn.widgets.MultiSelect(name="Highlight Categories", options=[], size=6)

def update_highlight_categories(event):
    factor = event.new
    if factor == "(None)":
        highlight_categories_widget.options = []
    else:
        if factor in merged_df.columns:
            cats = merged_df[factor].dropna().unique().tolist()
            cats = sorted(map(str, cats))
            highlight_categories_widget.options = cats
        else:
            highlight_categories_widget.options = []

highlight_factor_dropdown.param.watch(update_highlight_categories, "value")

# ============================================================
# 8. Panel Callback & Layout
# ============================================================
def make_initial_figure():
    return plot_pcoa(
        color_factor=color_dropdown.value if color_dropdown.value != "(None)" else None,
        shape_factor=shape_dropdown.value if shape_dropdown.value != "(None)" else None,
        show_names=show_names_checkbox.value,
        marker_size=marker_size_slider.value,
        zoom=zoom_slider.value,
        highlight_factor=None,
        highlight_cats=None,
        legend_marker_size=legend_marker_size_slider.value
    )

matplot_pane = pn.pane.Matplotlib(make_initial_figure(), tight=True)

@pn.depends(color_dropdown.param.value,
            shape_dropdown.param.value,
            show_names_checkbox.param.value,
            marker_size_slider.param.value,
            zoom_slider.param.value,
            legend_marker_size_slider.param.value,
            highlight_factor_dropdown.param.value,
            highlight_categories_widget.param.value)
def update_plot(color_factor, shape_factor, show_names, marker_size, zoom, legend_marker_size, highlight_factor, highlight_cats):
    if highlight_factor == "(None)":
        highlight_factor = None
    if color_factor == "(None)":
        color_factor = None
    if shape_factor == "(None)":
        shape_factor = None
    if not highlight_cats:
        highlight_cats = []
    fig = plot_pcoa(color_factor, shape_factor, show_names, marker_size, zoom, highlight_factor, highlight_cats, legend_marker_size=legend_marker_size)
    matplot_pane.object = fig
    return matplot_pane

# Button to save the current plot as PNG
save_button = pn.widgets.Button(name="Save Plot as PNG", button_type="primary")
def on_save(event):
    save_plot_as_png(
        filename="pcoa_plot.png",
        color_factor=color_dropdown.value if color_dropdown.value != "(None)" else None,
        shape_factor=shape_dropdown.value if shape_dropdown.value != "(None)" else None,
        show_names=show_names_checkbox.value,
        marker_size=marker_size_slider.value,
        zoom=zoom_slider.value,
        highlight_factor=highlight_factor_dropdown.value if highlight_factor_dropdown.value != "(None)" else None,
        highlight_cats=highlight_categories_widget.value,
        point_color=None,  # Change if you want a fixed color
        legend_marker_size=legend_marker_size_slider.value
    )
save_button.on_click(on_save)

app = pn.Column(
    pn.Row(
        pn.Column("<br/>**Normal Coloring**<br/>", color_dropdown, shape_dropdown),
        pn.Column("<br/>**Highlight**<br/>", highlight_factor_dropdown, highlight_categories_widget)
    ),
    pn.Row(show_names_checkbox, marker_size_slider, zoom_slider, legend_marker_size_slider),
    update_plot,
    save_button
)

pn.serve(app, start=True, show=True)

