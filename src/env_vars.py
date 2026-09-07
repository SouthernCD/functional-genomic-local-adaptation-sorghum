import os
import seaborn as sns
from Bio import Phylo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
import allel
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import geopandas as gpd
from shapely.geometry import box, MultiPolygon
from scipy.stats import pearsonr
from .tree import UnrootedTreePlotter
from .plot import set_plot_style
from .utils import mkdir
import networkx as nx    
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.patches import Circle



GROUP_COLOR_LOOKUP = {
    "AETHIOPICUM": "#000000",       # black
    "ARUNDINACEUM": "#999999",      # gray60
    "BICOLOR": "#FF0000",           # red
    "BICOLOR_CAUDATUM": "#8B7355",  # burlywood4
    "BICOLOR_DURRA": "#8B2323",     # brown4
    "BICOLOR_GUINEA": "#BDB76B",    # darkkhaki
    "BICOLOR_KAFIR": "#8B6969",     # rosybrown4
    "CAUDATUM": "#0000FF",          # blue
    "CAUDATUM_DURRA": "#66CDAA",    # aquamarine3
    "CAUDATUM_KAFIR": "#87CEEB",    # skyblue
    "DURRA": "#008B00",             # green4
    "GUINEA": "#FFA500",            # orange1
    "GUINEA_CAUDATUM": "#B3EE3A",   # olivedrab2
    "GUINEA_DURRA": "#8B814C",      # lightgoldenrod4
    "GUINEA_KAFIR": "#FF8247",      # sienna1
    "KAFIR": "#A020F0",             # purple
    "KAFIR_DURRA": "#FF69B4",       # hotpink
    "VERTICILLIFLORUM": "#737373",  # gray45
    "VIRGATUM": "#4D4D4D",          # gray30
    "Unknown": "#CCCCCC"            # gray80
}

PLOTBOTANICALTYPE_TO_GROUPID = {
    "Wild/weedy/unknown": "Unknown",
    "Guinea": "GUINEA",
    "Caudatum": "CAUDATUM",
    "Durra-caudatum": "CAUDATUM_DURRA",
    "Durra": "DURRA",
    "Durra-bicolor": "BICOLOR_DURRA",
    "Guinea-caudatum": "GUINEA_CAUDATUM",
    "Caudatum-bicolor": "BICOLOR_CAUDATUM",
    "Kafir": "KAFIR",
    "Kafir-caudatum": "CAUDATUM_KAFIR",
    "Guinea-bicolor": "BICOLOR_GUINEA",
    "Kafir-bicolor": "BICOLOR_KAFIR",
    "Guinea-kafir": "GUINEA_KAFIR",
    "Kafir-durra": "KAFIR_DURRA",
    "Guinea-durra": "GUINEA_DURRA",
    "Bicolor": "BICOLOR",
}


def nj_tree_plotter(nj_tree_file, landraces_reseq_df, save_fig_path):
    set_plot_style()

    plotbotanicaltype_color_lookup = {
        i: GROUP_COLOR_LOOKUP[j] for i, j in PLOTBOTANICALTYPE_TO_GROUPID.items()}

    landraces_reseq_df['PlotBotanicalType_Color'] = landraces_reseq_df.apply(
        lambda x: plotbotanicaltype_color_lookup.get(x['PlotBotanicalType'], 'gray'), axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    nj_plotter = UnrootedTreePlotter(Phylo.read(nj_tree_file, 'newick'))
    color_dict = landraces_reseq_df[['LIB', 'PlotBotanicalType_Color']].set_index(
        'LIB').to_dict()['PlotBotanicalType_Color']
    # color_dict中没有的设置为灰色
    for tip in nj_plotter.tree.get_terminals():
        if tip.name not in color_dict:
            color_dict[tip.name] = 'gray'

    nj_plotter.plot(clade_colors=color_dict, ax=ax, bg_lw=1)

    fig.savefig(save_fig_path, format='pdf', facecolor='none',
                edgecolor='none', bbox_inches='tight')

    plt.show()

    return


def landraces_botanical_map_plotter(landraces_reseq_df, save_fig_path, NATURAL_EARTH_SHP_FILE):
    set_plot_style()

    plotbotanicaltype_color_lookup = {
        i: GROUP_COLOR_LOOKUP[j] for i, j in PLOTBOTANICALTYPE_TO_GROUPID.items()}

    value_label = 'BotanicalType'
    title = 'Landrace Resequencing Samples in Africa'

    latitude_limit = (-37, 38)
    longitude_limit = (-20, 52)

    data = pd.DataFrame({
        'latitude': landraces_reseq_df['Latitude'],
        'longitude': landraces_reseq_df['Longitude'],
        'value': landraces_reseq_df['PlotBotanicalType'],
    })

    # # 把小于10的类别合并为一个类别
    # data = data[data['value'].notnull()]
    # data['value'] = data['value'].apply(lambda x: x if data['value'].value_counts()[x] > 4 else 'Others')

    fig, ax = plt.subplots(figsize=(8, 8))

    data = data[data['latitude'].notnull()]
    data = data[data['longitude'].notnull()]
    data = data[data['value'].notnull()]

    # 读取内置的世界地图数据
    world = gpd.read_file(NATURAL_EARTH_SHP_FILE)
    # 在 axes 对象上绘制地图
    world.boundary.plot(color='black', linewidth=0.5, ax=ax)

    # 设置地图边界
    ax.set_xlim(longitude_limit)  # 替换为你的左右边界的经度
    ax.set_ylim(latitude_limit)  # 替换为你的上下边界的纬度

    # scatter
    value_plot_order = sorted(data['value'].unique(), key=lambda x: len(
        data[data['value'] == x]), reverse=True)

    for value in value_plot_order:
        subdata = data[data['value'] == value]
        data_count = len(subdata)
        sns.scatterplot(
            x=subdata['longitude'],
            y=subdata['latitude'],
            color=plotbotanicaltype_color_lookup.get(
                value, 'gray') if plotbotanicaltype_color_lookup else None,
            label='%s (%d)' % (value, data_count),
            s=40,
            ax=ax,
        )
        # ax.scatter(
        #     subdata['longitude'], subdata['latitude'], c=[
        #         cmap(value_color)], label='%s (%d)' %
        #     (value, data_count), s=s)

    ax.set_title(title)

    # 显示图例，位置在右下角
    ax.legend(loc='lower right')

    # 为 landraces_reseq_df 添加一个新列，用于存储每个样本的绘图颜色
    landraces_reseq_df['PlotBotanicalType_Color'] = landraces_reseq_df.apply(
        lambda x: plotbotanicaltype_color_lookup.get(x['PlotBotanicalType'], 'gray'), axis=1)

    ax.set_title(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # 获取当前legend的handles和labels
    handles, labels = ax.get_legend_handles_labels()

    # 反转顺序
    handles = handles[::-1]
    labels = labels[::-1]

    # 设置 legend，调整字体和行间距
    ax.legend(handles, labels, loc='lower left',
              fontsize=11, labelspacing=0.3, frameon=False)
    plt.show()

    # save the figure
    fig.savefig(save_fig_path, format='pdf', facecolor='none',
                edgecolor='none', bbox_inches='tight')

    return


def env_vars_map_plotter(main_env,
                         small_envs,
                         landraces_reseq_df,
                         nj_tree_file,
                         env_info_df,
                         save_fig_path,
                         NATURAL_EARTH_SHP_FILE,
                         longitude_limit=(-20, 52),
                         latitude_limit=(-37, 22)):

    set_plot_style()

    # =========================
    # 1. 基本参数
    # =========================
    world = gpd.read_file(NATURAL_EARTH_SHP_FILE)

    # 只保留非洲，并裁剪到视窗范围
    africa = world[world["CONTINENT"] == "Africa"].copy()
    africa_bbox = box(longitude_limit[0], latitude_limit[0], 53, 38)
    africa = africa.clip(africa_bbox)

    # 去掉明显岛屿国家
    drop_names = {
        "Madagascar", "Mauritius", "Comoros", "Seychelles", "Cape Verde",
        "Sao Tome and Principe", "Reunion", "Mayotte"
    }
    africa = africa[~africa["NAME"].isin(drop_names)].copy()

    # 对每个国家保留最大 polygon，去掉零散小岛
    def keep_largest_part(geom):
        if geom is None or geom.is_empty:
            return geom
        if isinstance(geom, MultiPolygon):
            return max(geom.geoms, key=lambda g: g.area)
        return geom

    africa["geometry"] = africa["geometry"].apply(keep_largest_part)
    africa = africa[africa.geometry.notnull(
    ) & ~africa.geometry.is_empty].copy()

    env_desc_dict = dict(env_info_df[['env_id', 'short description']].values)
    env_unit_dict = dict(env_info_df[['env_id', 'units']].fillna('').values)
    env_scale_dict = dict(env_info_df[['env_id', 'scale']].values)

    # =========================
    # 2. 环境变量缩放 + 颜色映射
    # =========================
    def get_env_scale(env_name):
        scale = env_scale_dict.get(env_name, 1)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        return scale

    def get_scaled_env_series(env_name, df):
        scale = get_env_scale(env_name)
        return df[env_name].astype(float) / scale

    def get_env_cmap_norm(env_name, df):
        env_series = get_scaled_env_series(env_name, df)
        q_lo, q_hi = np.nanquantile(env_series, [0.02, 0.98])

        if env_name in ('striga', 'striga_crop'):
            cmap = plt.cm.viridis
            norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            if q_lo < 0 < q_hi:
                cmap = plt.cm.RdBu_r
                norm = mcolors.TwoSlopeNorm(vmin=q_lo, vcenter=0.0, vmax=q_hi)
            elif q_hi <= 0:
                cmap = plt.cm.viridis_r
                norm = mcolors.Normalize(vmin=q_lo, vmax=0.0)
            else:
                cmap = plt.cm.viridis
                norm = mcolors.Normalize(vmin=q_lo, vmax=q_hi)

        return cmap, norm

    def get_sample_colors(env_name, df):
        scaled_series = get_scaled_env_series(env_name, df)
        cmap, norm = get_env_cmap_norm(env_name, df)

        def val2rgba(v):
            return cmap(norm(v)) if np.isfinite(v) else (0.6, 0.6, 0.6, 1.0)

        color_list = [val2rgba(v) for v in scaled_series.values]
        return color_list, cmap, norm

    # =========================
    # 3. 只画地图（不画色带）
    # =========================

    def plot_env_map_only(
        ax,
        env_name,
        df,
        africa_gdf,
        longitude_limit=(-20, 50),
        latitude_limit=(-37, 30),
        point_size=28,
        edge_lw=0.5,
        show_title=False,
        title_size=11,
    ):
        point_colors, cmap, norm = get_sample_colors(env_name, df)

        ax.set_facecolor('white')
        ax.patch.set_alpha(1)

        africa_gdf.plot(
            ax=ax,
            facecolor='black',
            edgecolor='none',
            linewidth=0,
            zorder=0
        )

        africa_gdf.boundary.plot(
            ax=ax,
            color='0.6',
            linewidth=0.6,
            zorder=1
        )

        ax.set_xlim(longitude_limit)
        ax.set_ylim(latitude_limit)

        ax.scatter(
            df['Longitude'].values,
            df['Latitude'].values,
            s=point_size,
            c=point_colors,
            edgecolors='white',
            linewidths=edge_lw,
            zorder=2
        )

        # ax.axis('off')

        if show_title:
            ax.set_title(env_desc_dict.get(env_name, env_name),
                         fontsize=title_size, pad=4)

        return cmap, norm

    # =========================
    # 4. 单独画色带 + 左侧垂直标签 + 顶部单位
    # =========================

    def add_colorbar_to_axes(
        ax_parent,
        cmap,
        norm,
        label_text='',
        unit_text='',
        cbar_anchor=(0.08, 0.06, 0.05, 0.28),   # x, y, w, h
        cbar_ticksize=8,
        label_size=10,
        unit_size=9,
        unit_pos='top',   # 'top' or 'bottom'
        label_pos='left',  # 'left' or 'right' or 'top' or 'bottom'
        panel_pad=(0.02, 0.02, 0.02, 0.02),
        label_gap=0.025,
        unit_gap=0.015,
        border_color='0.25',
        border_lw=0.8,
    ):
        cb_x, cb_y, cb_w, cb_h = cbar_anchor
        pad_l, pad_b, pad_r, pad_t = panel_pad

        panel = Rectangle(
            (cb_x - pad_l, cb_y - pad_b),
            cb_w + pad_l + pad_r,
            cb_h + pad_b + pad_t,
            transform=ax_parent.transAxes,
            facecolor='white',
            edgecolor='none',
            linewidth=0,
            zorder=50
        )
        panel.set_in_layout(False)
        ax_parent.add_patch(panel)

        ax_cbar = inset_axes(
            ax_parent,
            width="100%",
            height="100%",
            bbox_to_anchor=(cb_x, cb_y, cb_w, cb_h),
            bbox_transform=ax_parent.transAxes,
            loc='lower left',
            borderpad=0
        )
        ax_cbar.set_facecolor('white')
        ax_cbar.patch.set_alpha(1)
        ax_cbar.set_zorder(51)
        ax_cbar.set_in_layout(False)

        cb = ColorbarBase(
            ax_cbar,
            cmap=cmap,
            norm=norm,
            orientation='vertical'
        )
        cb.locator = MaxNLocator(nbins=4)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=cbar_ticksize, length=2.5, width=0.8)

        for spine in ax_cbar.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(border_lw)
            spine.set_edgecolor(border_color)

        if label_pos == 'left':
            txt1 = ax_parent.text(
                cb_x - label_gap,
                cb_y + cb_h / 2,
                label_text,
                rotation=90,
                va='center',
                ha='center',
                fontsize=label_size,
                transform=ax_parent.transAxes,
                zorder=52,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor='none')
            )
        elif label_pos == 'right':
            txt1 = ax_parent.text(
                cb_x + cb_w + label_gap,
                cb_y + cb_h / 2,
                label_text,
                rotation=-90,
                va='center',
                ha='center',
                fontsize=label_size,
                transform=ax_parent.transAxes,
                zorder=52,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor='none')
            )
        elif label_pos == 'top':
            txt1 = ax_parent.text(
                cb_x + cb_w / 2,
                cb_y + cb_h + label_gap,
                label_text,
                va='bottom',
                ha='center',
                fontsize=label_size,
                transform=ax_parent.transAxes,
                zorder=52,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor='none')
            )
        else:  # 'bottom'
            txt1 = ax_parent.text(
                cb_x + cb_w / 2,
                cb_y - label_gap,
                label_text,
                va='top',
                ha='center',
                fontsize=label_size,
                transform=ax_parent.transAxes,
                zorder=52,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor='none')
            )
        txt1.set_in_layout(False)

        unit_text = '' if unit_text is None else str(unit_text).strip()
        if unit_text and unit_text.lower() not in {'nan', 'none'}:
            if unit_pos == 'top':
                txt2 = ax_parent.text(
                    cb_x + cb_w / 2,
                    cb_y + cb_h + unit_gap,
                    unit_text,
                    va='bottom',
                    ha='center',
                    fontsize=unit_size,
                    transform=ax_parent.transAxes,
                    zorder=52
                )
            else:
                txt2 = ax_parent.text(
                    cb_x + cb_w / 2,
                    cb_y - unit_gap,
                    unit_text,
                    va='top',
                    ha='center',
                    fontsize=unit_size,
                    transform=ax_parent.transAxes,
                    zorder=52
                )
            txt2.set_in_layout(False)

        return ax_cbar

    # =========================
    # 5. 树
    # =========================

    def add_tree_inset(
        ax,
        env_name,
        df,
        nj_tree_file,
        bbox_to_anchor=(-0.08, -0.02, 1, 1),
        width="62%",
        height="62%",
        rotate_deg=-60.0,
    ):
        point_colors, _, _ = get_sample_colors(env_name, df)
        sample_names = df['LIB'].tolist()
        color_dict = dict(zip(sample_names, point_colors))

        ax_tree = inset_axes(
            ax,
            width=width,
            height=height,
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=ax.transAxes,
            loc='lower left'
        )

        ax_tree.set_facecolor('white')
        ax_tree.patch.set_alpha(1)
        ax_tree.axis('off')
        ax_tree.set_aspect('equal')
        ax_tree.set_in_layout(False)

        nj_plotter = UnrootedTreePlotter(Phylo.read(nj_tree_file, 'newick'))
        nj_plotter.plot(
            ax=ax_tree,
            clade_colors=color_dict,
            bg_lw=1,
            use_edge_length=True,
            rotate_tree=np.deg2rad(rotate_deg)
        )

        return ax_tree

    # =========================
    # 6. 主图
    # =========================
    fig = plt.figure(figsize=(9, 10), facecolor='white')
    ax_main = fig.add_subplot(111)
    ax_main.set_facecolor('white')
    ax_main.patch.set_alpha(1)
    ax_main.set_in_layout(False)

    main_cmap, main_norm = plot_env_map_only(
        ax=ax_main,
        env_name=main_env,
        df=landraces_reseq_df,
        africa_gdf=africa,
        longitude_limit=longitude_limit,
        latitude_limit=latitude_limit,
        point_size=46,
        edge_lw=0.6,
        show_title=False
    )

    add_colorbar_to_axes(
        ax_parent=ax_main,
        cmap=main_cmap,
        norm=main_norm,
        label_text=env_desc_dict.get(main_env, main_env),
        unit_text=env_unit_dict.get(main_env, ''),
        cbar_anchor=(0.25, 0.46, 0.03, 0.16),
        cbar_ticksize=9,
        label_size=14,
        unit_size=11,
        panel_pad=(0.050, 0.015, 0.015, 0.015),
        label_gap=0.022,
        unit_gap=0.01,
        border_lw=0.8,
        unit_pos='bottom',
        label_pos='top'
    )

    add_tree_inset(
        ax=ax_main,
        env_name=main_env,
        df=landraces_reseq_df,
        nj_tree_file=nj_tree_file,
        bbox_to_anchor=(-0.12, -0.02, 1, 1),
        width="62%",
        height="62%",
        rotate_deg=-60.0
    )

    # =========================
    # 7. 小图
    # =========================
    y0 = 0.95
    h = 0.270
    x0 = 0.04
    w_total = 0.77

    gap = 0.001
    w = (w_total - 2 * gap) / 3.0

    # top_bg = Rectangle(
    #     (x0, y0),
    #     w_total, h,
    #     transform=ax_main.transAxes,
    #     facecolor='white',
    #     edgecolor='none',
    #     linewidth=0,
    #     zorder=19
    # )
    # top_bg.set_in_layout(False)
    # ax_main.add_patch(top_bg)

    small_boxes = [
        (x0,                 y0, w, h),
        (x0 + w + gap,       y0, w, h),
        (x0 + 2*(w + gap),   y0, w, h),
    ]

    for env_name, box_ in zip(small_envs, small_boxes):
        x, y, w_box, h_box = box_

        ax_in = inset_axes(
            ax_main,
            width="100%",
            height="100%",
            bbox_to_anchor=(x, y, w_box, h_box),
            bbox_transform=ax_main.transAxes,
            loc='lower left',
            borderpad=0,
        )
        ax_in.set_facecolor('white')
        ax_in.patch.set_alpha(1)
        ax_in.set_zorder(21)
        ax_in.set_in_layout(False)

        cmap_small, norm_small = plot_env_map_only(
            ax=ax_in,
            env_name=env_name,
            df=landraces_reseq_df,
            africa_gdf=africa,
            longitude_limit=longitude_limit,
            latitude_limit=latitude_limit,
            point_size=14,
            edge_lw=0.35,
            show_title=False
        )

        add_colorbar_to_axes(
            ax_parent=ax_in,
            cmap=cmap_small,
            norm=norm_small,
            label_text="",
            unit_text=env_unit_dict.get(env_name, ''),
            cbar_anchor=(0.1, 0.1, 0.085, 0.44),
            cbar_ticksize=7,
            label_size=11,
            unit_size=9,
            panel_pad=(0.05, 0.015, 0.015, 0.015),
            label_gap=0.060,
            unit_gap=0.01,
            border_lw=0.8
        )

        for spine in ax_in.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor('0.25')

            ax_in.set_xticks([])
            ax_in.set_yticks([])
            ax_in.tick_params(left=False, bottom=False,
                              labelleft=False, labelbottom=False)

        ax_in.set_title(
            env_desc_dict.get(env_name, env_name),
            loc='left',     # 关键
            fontsize=12,
            pad=4
        )

    ax_main.axis('off')

    # fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    fig.savefig(save_fig_path, format='pdf',
                facecolor='white', edgecolor='none')
    # print(f"Figure saved to {save_fig_path}")
    plt.show()

    return


def rda_analysis(vcf, landraces_env_df, kinship_file, output_dir):
    mkdir(output_dir)

    # 0) 环境矩阵：去缺失并固定样本顺序
    env = landraces_env_df.copy()
    env_clean = env.dropna()
    keep_ids = env_clean.index.tolist()
    print("env_clean:", env_clean.shape)

    # 1) 读 VCF，并对齐样本顺序
    callset = allel.read_vcf(vcf, samples=keep_ids)   # 直接只读保留样本（最稳）
    samples_vcf = callset['samples'].tolist()
    assert samples_vcf == keep_ids, "VCF样本顺序与env不一致"

    gt = allel.GenotypeArray(callset['calldata/GT'])
    gn = gt.to_n_alt().T.astype("float32")   # (n_samples, n_variants), 0/1/2, 缺失为 -1
    print("gn raw:", gn.shape)

    # 2) 处理缺失：先把 -1 变为 NaN，再按位点做“均值插补”
    gn[gn < 0] = np.nan
    imp = SimpleImputer(strategy="mean")
    gn_imp = imp.fit_transform(gn)           # 每一列用非缺失均值填充
    # 若某列全缺失，SimpleImputer 会报错；为防万一，先移除全缺失列：
    all_nan_cols = np.isnan(gn).all(axis=0)
    if all_nan_cols.any():
        gn = gn[:, ~all_nan_cols]
        gn_imp = imp.fit_transform(gn)

    # 3) 去掉在当前样本集合上仍然单态的位点（方差=0）
    var = np.nanvar(gn_imp, axis=0)
    gn_imp = gn_imp[:, var > 0]
    print("gn cleaned:", gn_imp.shape)

    # 4) 基因型中心化
    X = gn_imp - np.mean(gn_imp, axis=0)

    # 5) 环境矩阵标准化（用去缺失后的 env_clean）
    E = StandardScaler().fit_transform(env_clean.values)

    # 6) kinship 对齐样本顺序
    K = pd.read_csv(kinship_file, sep=',', names=landraces_env_df.index)
    K.index = landraces_env_df.index
    K = K.loc[keep_ids, keep_ids].values
    print("K:", K.shape)

    # 7) 最后做一次健壮性检查
    assert np.isfinite(X).all(), "X 仍有 NaN/Inf"
    assert np.isfinite(E).all(), "E 仍有 NaN/Inf"
    assert np.isfinite(K).all(), "K 仍有 NaN/Inf"

    # 8) RDA 分析

    # 拟合基因型对环境
    lm = LinearRegression().fit(E, X)
    X_hat = lm.predict(E)

    # PCA on fitted values
    pca = PCA()
    rda_scores = pca.fit_transform(X_hat)

    # RDA 结果
    explained = pca.explained_variance_ratio_
    print("Explained variance:", explained[:5])

    # 计算 R2_adj
    SS_total = np.sum((X - X.mean(axis=0))**2)
    SS_explained = np.sum((X_hat - X.mean(axis=0))**2)
    R2 = SS_explained / SS_total
    n = X.shape[0]
    p = E.shape[1]                  # 环境变量个数（37）
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    print("Adjusted R2:", R2_adj)

    rda1_contrib = []
    rda2_contrib = []

    for j in range(E.shape[1]):
        r1, _ = pearsonr(E[:, j], rda_scores[:, 0])
        r2, _ = pearsonr(E[:, j], rda_scores[:, 1])
        rda1_contrib.append(r1)
        rda2_contrib.append(r2)

    rda1_contrib = np.array(rda1_contrib)
    rda2_contrib = np.array(rda2_contrib)

    rda_samples_df = pd.DataFrame(rda_scores[:, :2], index=keep_ids, columns=['RDA1', 'RDA2'])
    rda_env_contrib_df = pd.DataFrame({'env_id': env_clean.columns, 'RDA1_contrib': rda1_contrib, 'RDA2_contrib': rda2_contrib})
    rda_explained_df = pd.DataFrame({'RDA': [f'RDA{i+1}' for i in range(len(explained))], 'explained_variance': explained})
    rda_explained_df = pd.concat([rda_explained_df, pd.DataFrame({'RDA': ['R2_adj'], 'explained_variance': [R2_adj]})], ignore_index=True)

    rda_samples_df.to_csv(os.path.join(output_dir, "rda_samples.csv"))
    rda_env_contrib_df.to_csv(os.path.join(output_dir, "rda_env_contrib.csv"))
    rda_explained_df.to_csv(os.path.join(output_dir, "rda_explained.csv"))

    return

def plot_rda_biplot(rda_samples_df, rda_env_contrib_df, rda_explained_df, env_info_df, save_fig_path):

    cmap = plt.get_cmap('Dark2')

    category_list = env_info_df['category'].unique()
    category_color_dict = {cat: cmap(i) for i, cat in enumerate(category_list)}

    env_info_df['category_color'] = env_info_df['category'].map(category_color_dict)
    env_info_df

    rda1_contrib = rda_env_contrib_df['RDA1_contrib'].values
    rda2_contrib = rda_env_contrib_df['RDA2_contrib'].values
    rda1_sample_scores = rda_samples_df['RDA1'].values
    rda2_sample_scores = rda_samples_df['RDA2'].values

    # ====== Step 5: 绘制双标图（biplot）======
    fig, ax = plt.subplots(1, 1, figsize=(5.1, 5))

    # 绘制样品点
    sns.scatterplot(x=rda1_sample_scores, y=rda2_sample_scores, s=80, color='black', ax=ax, alpha=0.7)

    label_dict = env_info_df[['env_id', 'env_abbr']].set_index('env_id')['env_abbr'].to_dict()
    label_color_dict = env_info_df.set_index('env_id')['category_color'].to_dict()

    # 变量箭头（以原点为起点）
    for i, feature in enumerate(rda_env_contrib_df['env_id'].values):
        x_arrow = rda1_contrib[i] * 300 # 放大长度，便于可视化
        y_arrow = rda2_contrib[i] * 300
        ax.arrow(0, 0, x_arrow, y_arrow, color='gray', alpha=0.6, head_width=10, width=0.05)

        # ===== 关键：沿箭头方向平移 =====
        vec = np.array([x_arrow, y_arrow])
        norm = np.linalg.norm(vec)

        if norm == 0:
            continue

        direction = vec / norm   # 单位向量

        offset = 40  # 控制 label 离箭头的距离（自己调）
        x_text = x_arrow + direction[0] * offset
        y_text = y_arrow + direction[1] * offset

        ax.text(x_text, y_text,
                label_dict[feature],
                color=label_color_dict.get(feature, 'black'),
                fontsize=10,
                fontweight='bold',
                ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

        # ax.text(x_arrow + 30, y_arrow + 30, label_dict[feature], color=label_color_dict.get(feature, 'black'), fontsize=10, fontweight='bold', ha='center', va='center')

    # 图形美化
    rda1_explained = rda_explained_df.loc[rda_explained_df['RDA'] == 'RDA1', 'explained_variance'].values[0]
    rda2_explained = rda_explained_df.loc[rda_explained_df['RDA'] == 'RDA2', 'explained_variance'].values[0]
    r2_adj = rda_explained_df.loc[rda_explained_df['RDA'] == 'R2_adj', 'explained_variance'].values[0]
    ax.set_xlabel(f"RDA1 ({rda1_explained*r2_adj*100:.2f}%)")
    ax.set_ylabel(f"RDA2 ({rda2_explained*r2_adj*100:.2f}%)")
    # plt.title("PCA Biplot of Environmental Variables")
    # plt.grid(True)
    ax.axhline(0, linestyle='dotted', color='gray')
    ax.axvline(0, linestyle='dotted', color='gray')

    # 关闭坐标轴
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


    ax.set_xlim(ax.get_xlim()[0]*1.2, ax.get_xlim()[1])
    fig.patch.set_facecolor('none')
    ax.patch.set_alpha(0)

    plt.tight_layout()

    fig.savefig(save_fig_path, format='pdf', facecolor='none',
                    edgecolor='none')
    plt.show()    



def plot_env_corr_network(env_info_df, landraces_env_df, r_threshold=0.7, corr_file=None, save_fig_path=None):

    # =========================
    # 工具函数
    # =========================
    def orient_flat(pos_dict, flatten=0.65, align='x'):
        nodes = list(pos_dict.keys())
        XY = np.array([pos_dict[n] for n in nodes], dtype=float)

        if len(nodes) == 1:
            return {nodes[0]: np.array([0.0, 0.0])}

        XY -= XY.mean(axis=0)
        _, _, Vt = np.linalg.svd(XY, full_matrices=False)
        XY = XY @ Vt.T

        if align == 'y':
            XY = XY[:, ::-1]

        XY[:, 1] *= flatten

        max_abs = np.abs(XY).max()
        if max_abs > 0:
            XY /= max_abs

        return {n: XY[i] for i, n in enumerate(nodes)}


    def scale_pos(pos, sx=1.0, sy=1.0, center=True):
        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        cx, cy = (xs.mean(), ys.mean()) if center else (0.0, 0.0)

        new = {}
        for n, (x, y) in pos.items():
            new[n] = np.array([cx + (x - cx) * sx, cy + (y - cy) * sy])
        return new


    def normalize_to_box(pos, box_w=1.0, box_h=1.0, keep_aspect=True):
        nodes = list(pos.keys())
        XY = np.array([pos[n] for n in nodes], dtype=float)

        if len(nodes) == 1:
            return {nodes[0]: np.array([0.0, 0.0])}

        min_xy = XY.min(axis=0)
        max_xy = XY.max(axis=0)
        span = max_xy - min_xy
        span[span == 0] = 1.0

        XY = XY - (min_xy + max_xy) / 2

        if keep_aspect:
            s = max(span[0] / box_w, span[1] / box_h)
            if s == 0:
                s = 1.0
            XY = XY / s
        else:
            XY[:, 0] = XY[:, 0] / span[0] * box_w
            XY[:, 1] = XY[:, 1] / span[1] * box_h

        return {n: XY[i] for i, n in enumerate(nodes)}


    def translate_pos(pos, dx=0.0, dy=0.0):
        return {n: np.array([xy[0] + dx, xy[1] + dy]) for n, xy in pos.items()}


    def build_corr_graph(df, method='pearson', r_threshold=0.7, corr_file=None):
        C = df.corr(method=method)

        if corr_file is not None:
            C.to_csv(corr_file)
            # print(f"Correlation matrix saved to: {corr_file}")

        edges = (
            C.stack()
            .reset_index()
            .rename(columns={'level_0': 'u', 'level_1': 'v', 0: 'r'})
        )
        edges = edges[edges['u'] < edges['v']]
        edges = edges[edges['r'].abs() >= r_threshold]

        G = nx.Graph()
        G.add_nodes_from(df.columns)
        for _, row in edges.iterrows():
            G.add_edge(row['u'], row['v'], r=row['r'], strength=abs(row['r']))
        return G


    def get_connected_components_and_isolates(G):
        comps = [G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) > 1]
        isolates = sorted([n for n, d in G.degree() if d == 0])
        comps.sort(key=lambda H: (H.number_of_nodes(), H.number_of_edges()), reverse=True)
        return comps, isolates


    def get_component_layout(H, layout='spring', seed=42, k=None, iterations=500):
        nV = H.number_of_nodes()
        nE = H.number_of_edges()

        if nV == 1:
            node = list(H.nodes())[0]
            return {node: np.array([0.0, 0.0])}

        if layout == 'spring':
            k_use = k if k is not None else 4.8 / max(np.sqrt(nV), 1.0)
            pos = nx.spring_layout(
                H,
                seed=seed,
                weight='strength',
                k=k_use,
                iterations=iterations
            )
        else:
            if nE > 0:
                L = {(u, v): 1.0 / max(d.get('strength', 1e-6), 1e-6)
                    for u, v, d in H.edges(data=True)}
                pos = nx.kamada_kawai_layout(H, dist=L)
            else:
                pos = nx.spring_layout(H, seed=seed)

        pos = orient_flat(pos, flatten=0.70, align='x')
        pos = scale_pos(pos, sx=1.0, sy=1.10, center=True)
        return pos


    def get_node_size_map(G, degree_mode='count', node_size_min=400, node_size_max=2000):
        if degree_mode == 'strength' and G.number_of_edges() > 0:
            val_by_node = {n: 0.0 for n in G.nodes()}
            for u, v, d in G.edges(data=True):
                w = abs(d.get('r', 0.0))
                val_by_node[u] += w
                val_by_node[v] += w
        else:
            val_by_node = dict(G.degree())

        vals = np.array([val_by_node[n] for n in G.nodes()], dtype=float)
        if np.isclose(vals.max(), vals.min()):
            return {n: (node_size_min + node_size_max) / 2 for n in G.nodes()}

        interp_sizes = np.interp(
            vals,
            (vals.min(), vals.max()),
            (node_size_min, node_size_max)
        )
        return {n: s for n, s in zip(G.nodes(), interp_sizes)}


    def abbr_to_env_id(abbr_or_id, env_info_df):
        if abbr_or_id in set(env_info_df["env_id"]):
            return abbr_or_id

        abbr_map = dict(zip(env_info_df["env_abbr"], env_info_df["env_id"]))
        if abbr_or_id in abbr_map:
            return abbr_map[abbr_or_id]

        return None


    # =========================
    # 专用绘图函数：2节点连通图
    # =========================
    def draw_two_node_component(
        ax,
        node_top,
        node_bottom,
        top_center,
        edge_r,
        label_dict,
        node_size_map,
        cmap,
        edge_norm,
        text_color_dict=None,
        vertical_gap=0.85,
        fixed_node_size=None,
        radius_scale=110.0,
        node_facecolor="#F7F7F7",
        node_edgecolor="#5a7aa8",
        node_linewidth=0.8,
        font_size=11,
        font_family='Arial',
        font_weight='bold',
    ):
        x_top, y_top = top_center
        x_bottom, y_bottom = x_top, y_top - vertical_gap

        size_top = fixed_node_size if fixed_node_size is not None else node_size_map[node_top]
        size_bottom = fixed_node_size if fixed_node_size is not None else node_size_map[node_bottom]

        r_top = np.sqrt(size_top) / radius_scale
        r_bottom = np.sqrt(size_bottom) / radius_scale

        edge_color = cmap(edge_norm(edge_r))

        y1 = y_top - r_top
        y2 = y_bottom + r_bottom
        ax.plot([x_top, x_bottom], [y1, y2], color=edge_color, linewidth=2.4, zorder=1)

        circ_top = Circle(
            (x_top, y_top), radius=r_top,
            facecolor=node_facecolor, edgecolor=node_edgecolor,
            linewidth=node_linewidth, zorder=2
        )
        circ_bottom = Circle(
            (x_bottom, y_bottom), radius=r_bottom,
            facecolor=node_facecolor, edgecolor=node_edgecolor,
            linewidth=node_linewidth, zorder=2
        )
        ax.add_patch(circ_top)
        ax.add_patch(circ_bottom)

        top_text_color = text_color_dict.get(node_top, 'black') if text_color_dict is not None else 'black'
        bottom_text_color = text_color_dict.get(node_bottom, 'black') if text_color_dict is not None else 'black'

        ax.text(
            x_top, y_top, label_dict.get(node_top, node_top),
            ha='center', va='center',
            fontsize=font_size, fontfamily=font_family,
            fontweight=font_weight, color=top_text_color, zorder=3
        )
        ax.text(
            x_bottom, y_bottom, label_dict.get(node_bottom, node_bottom),
            ha='center', va='center',
            fontsize=font_size, fontfamily=font_family,
            fontweight=font_weight, color=bottom_text_color, zorder=3
        )


    # =========================
    # 专用绘图函数：独立节点
    # =========================
    def draw_isolated_nodes(
        ax,
        node_positions,
        label_dict,
        node_size_map,
        text_color_dict=None,
        fixed_node_size=None,
        radius_scale=110.0,
        node_facecolor="#F7F7F7",
        node_edgecolor="#5a7aa8",
        node_linewidth=0.8,
        font_size=11,
        font_family='Arial',
        font_weight='bold',
    ):
        for node, (x, y) in node_positions.items():
            size = fixed_node_size if fixed_node_size is not None else node_size_map[node]
            radius = np.sqrt(size) / radius_scale

            circ = Circle(
                (x, y), radius=radius,
                facecolor=node_facecolor, edgecolor=node_edgecolor,
                linewidth=node_linewidth, zorder=2
            )
            ax.add_patch(circ)

            text_color = text_color_dict.get(node, 'black') if text_color_dict is not None else 'black'

            ax.text(
                x, y, label_dict.get(node, node),
                ha='center', va='center',
                fontsize=font_size, fontfamily=font_family,
                fontweight=font_weight, color=text_color, zorder=3
            )


    # =========================
    # 调用逻辑：这张图的具体布局
    # =========================
    edgecolors = '#525A61'
    label_font_size = 12

    label_dict = {
        env: env_info_df.set_index('env_id').at[env, 'env_abbr']
        for env in landraces_env_df.columns
    }

    # 文字颜色：按 env_info_df['category_color']
    text_color_dict = env_info_df.set_index('env_id')['category_color'].to_dict()

    env = landraces_env_df.copy()
    env_clean = env.dropna()

    G = build_corr_graph(
        env_clean,
        method='pearson',
        r_threshold=r_threshold,
        corr_file=corr_file
    )

    comps, isolates = get_connected_components_and_isolates(G)

    H0 = comps[0]
    small_two_node_comps = comps[1:]
    global_pos = {}

    # 1. 最大连通图：更松弛 + 盒子更大一些
    pos0 = get_component_layout(H0, layout='spring', seed=42, iterations=700, k=1.15)
    pos0 = normalize_to_box(pos0, box_w=7.0, box_h=6.6, keep_aspect=True)
    pos0 = scale_pos(pos0, sx=0.8, sy=1.4, center=True)
    pos0 = translate_pos(pos0, dx=0.0, dy=1.52)
    global_pos.update(pos0)

    # 2. 节点大小
    node_size_map = get_node_size_map(
        G,
        degree_mode='count',
        node_size_min=400,
        node_size_max=2000
    )

    # 3. 给小图统一一个固定节点大小
    small_nodes = []
    for H in small_two_node_comps:
        if H.number_of_nodes() == 2 and H.number_of_edges() == 1:
            small_nodes.extend(list(H.nodes()))

    if len(small_nodes) > 0:
        fixed_small_node_size = float(np.median([node_size_map[n] for n in small_nodes]))
    else:
        fixed_small_node_size = 700.0

    # 手动调这个，小图圈圈大小
    SMALL_NODE_RADIUS_SCALE = 150

    # 4. 颜色映射
    cmap_name = 'coolwarm'
    cmap = plt.get_cmap(cmap_name)
    edge_norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=edge_norm
    )
    sm.set_array([])

    shift_x = -0.1
    shift_y = 0.5
    x_gap = 0.4

    # 5. 小连通图位置
    two_node_specs = [
        {"top_center": (-0.95 + shift_x, -0.02 + shift_y), "vertical_gap": 0.72},
        {"top_center": (-0.95 + x_gap + shift_x, -0.02 + shift_y), "vertical_gap": 0.72},
        {"top_center": (-0.95 + 2 * x_gap + shift_x, -0.02 + shift_y), "vertical_gap": 0.72},
    ]

    # 6. 独立节点位置
    isolated_node_positions_input = {
        "WTD": (-0.95 + 3 * x_gap + shift_x, -0.02 + shift_y),
        "PHS": (-0.95 + 3 * x_gap + shift_x, -0.74 + shift_y),
        "PDQ": (-0.95 + 4 * x_gap + shift_x, -0.02 + shift_y),
        "PH":  (-0.95 + 4 * x_gap + shift_x, -0.74 + shift_y),
        "SMC": (-0.95 + 5 * x_gap + shift_x, -0.02 + shift_y),
    }

    isolated_node_positions = {}
    unresolved_keys = []

    for key, xy in isolated_node_positions_input.items():
        env_id = abbr_to_env_id(key, env_info_df)
        if env_id is None:
            unresolved_keys.append(key)
        else:
            isolated_node_positions[env_id] = xy

    if unresolved_keys:
        print("这些 isolated key 无法匹配到 env_id 或 env_abbr：", unresolved_keys)

    isolated_node_positions = {
        n: xy for n, xy in isolated_node_positions.items() if n in isolates
    }

    print("isolates =", isolates)
    print("isolated_node_positions after mapping/filter =", isolated_node_positions)

    # 7. 画图
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.patch.set_alpha(0)
    ax.axis('off')

    # 7.1 大连通图边
    if H0.number_of_edges() > 0:
        edgelist0 = list(H0.edges())
        r_vals0 = np.array([H0[u][v]['r'] for u, v in edgelist0])
        widths0 = np.clip(np.abs(r_vals0) * 3.0, 0.8, 8.0)

        nx.draw_networkx_edges(
            H0, global_pos, ax=ax,
            edgelist=edgelist0,
            edge_color=r_vals0,
            edge_cmap=cmap,
            edge_vmin=-1, edge_vmax=1,
            width=widths0,
            alpha=0.9
        )

    # 7.2 大连通图节点
    nx.draw_networkx_nodes(
        H0, global_pos, ax=ax,
        nodelist=list(H0.nodes()),
        node_size=[node_size_map[n] for n in H0.nodes()],
        node_color="#F7F7F7",
        linewidths=0.8,
        edgecolors=edgecolors
    )

    # 7.3 大连通图标签
    for n, (x, y) in global_pos.items():
        ax.text(
            x, y,
            label_dict.get(n, n),
            ha='center', va='center',
            fontsize=label_font_size,
            fontfamily='Arial',
            fontweight='bold',
            color=text_color_dict.get(n, 'black'),
            zorder=3
        )

    # 7.4 二节点连通图
    for i, H in enumerate(small_two_node_comps):
        if H.number_of_nodes() != 2 or H.number_of_edges() != 1:
            continue
        if i >= len(two_node_specs):
            continue

        nodes = list(H.nodes())
        u, v = nodes[0], nodes[1]
        edge_r = H[u][v]['r']

        draw_two_node_component(
            ax=ax,
            node_top=u,
            node_bottom=v,
            top_center=two_node_specs[i]["top_center"],
            edge_r=edge_r,
            label_dict=label_dict,
            node_size_map=node_size_map,
            cmap=cmap,
            edge_norm=edge_norm,
            text_color_dict=text_color_dict,
            vertical_gap=two_node_specs[i].get("vertical_gap", 0.95),
            fixed_node_size=fixed_small_node_size,
            radius_scale=SMALL_NODE_RADIUS_SCALE,
            node_facecolor="#F7F7F7",
            node_edgecolor=edgecolors,
            node_linewidth=0.8,
            font_size=label_font_size,
            font_family='Arial',
            font_weight='bold',
        )

    # 7.5 独立节点
    draw_isolated_nodes(
        ax=ax,
        node_positions=isolated_node_positions,
        label_dict=label_dict,
        node_size_map=node_size_map,
        text_color_dict=text_color_dict,
        fixed_node_size=fixed_small_node_size,
        radius_scale=SMALL_NODE_RADIUS_SCALE,
        node_facecolor="#F7F7F7",
        node_edgecolor=edgecolors,
        node_linewidth=0.8,
        font_size=label_font_size,
        font_family='Arial',
        font_weight='bold',
    )

    # 7.6 色带
    cax = fig.add_axes([0.45, 0.435, 0.14, 0.03])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    cbar.set_label("r", fontsize=14, loc='center')
    cbar.ax.tick_params(labelsize=11, axis='both', which='major', width=0.8)
    cbar.outline.set_linewidth(0.8)

    ax.set_aspect('equal')

    fig.savefig(
        save_fig_path,
        format='pdf',
        transparent=True,
        edgecolor='none',
    )

    plt.show()    