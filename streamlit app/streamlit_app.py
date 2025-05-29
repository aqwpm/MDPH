import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
import pyreadr
from tqdm import tqdm
import geopandas as gpd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import re


# 你的颜色变量
BLEU = "#000091"
ROUGE = "#E1000F"
PALETTE = ["#ed6a5a", "#9bc1bc", "#f4f1bb"]

def big_title(text):
    st.markdown(f'<span style="font-size:2.8em; color:{BLEU}; font-weight:bold;">{text}</span>', unsafe_allow_html=True)

def main_title(text):
    st.markdown(f'<span style="font-size:2.2em; color:{BLEU}; font-weight:bold;">{text}</span>', unsafe_allow_html=True)

def sub_title(text):
    st.markdown(f'<span style="font-size:1.5em; color:{BLEU}; font-weight:bold;">{text}</span>', unsafe_allow_html=True)


# 最大标题
big_title("MDPH")
st.write("L’objectif était d’identifier des typologies de fonctionnement à partir des données du baromètre CNSA 2022, afin de mieux comprendre les différences de performance, de délais du traitement moyens et de satisfaction des usagers.")

# 一级大标题
main_title("1. Data")

st.markdown("""
- Baromètre des maisons départementales des personnes handicapées :

https://www.cnsa.fr/vous-etes-une-personne-handicapee-ou-un-proche/barometre-des-maisons-departementales-des-personnes-handicapees

- Sur la base des données communiquées par la CNSA, les données de RH, de taux d’accord

- Données de l’insee Statistiques locales
""")

st.markdown("<br>", unsafe_allow_html=True)

main_title("2. Nettoyage de données")
sub_title("2.1 Données sur l’activité")

st.divider()

DATA_DIR = Path('baromètre/brutes')

# ===== 展示用代码块 Start =====
with st.expander("Afficher le code de nettoyage (Show code)"):
    st.code("""
def load_activite():
    df = pd.read_excel(DATA_DIR / 'activite.xlsx', skiprows=2, nrows=108)
    df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
    df['nom_département'] = df['année'].str.split(' -', n=1).str[1]
    df['département'] = df['année'].str.split(' -', n=1).str[0].str.strip()
    df['département'] = df.apply(
        lambda row: '69M' if row['nom_département'] == 'Métropole de Lyon' else row['département'],
        axis=1
    )
    value_vars = [col for col in df.columns if col in ['2019', '2020', '2021', '2022']]
    df_long = df.melt(id_vars=['département', 'nom_département'],
                      value_vars=value_vars,
                      var_name='année',
                      value_name='nombre_décisions_avis_rendus')
    df_long = df_long.iloc[2:].reset_index(drop=True)
    return df_long
    """, language='python')
# ===== 展示用代码块 End =====

# ===== 实际运行用 =====
@st.cache_data
def load_activite():
    df = pd.read_excel(DATA_DIR / 'activite.xlsx', skiprows=2, nrows=108)
    df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
    df['nom_département'] = df['année'].str.split(' -', n=1).str[1]
    df['département'] = df['année'].str.split(' -', n=1).str[0].str.strip()
    df['département'] = df.apply(
        lambda row: '69M' if row['nom_département'] == 'Métropole de Lyon' else row['département'],
        axis=1
    )
    value_vars = [col for col in df.columns if col in ['2019', '2020', '2021', '2022']]
    df_long = df.melt(id_vars=['département', 'nom_département'],
                      value_vars=value_vars,
                      var_name='année',
                      value_name='nombre_décisions_avis_rendus')
    df_long = df_long.iloc[2:].reset_index(drop=True)
    return df_long

activite = load_activite()
st.write(activite.head())

main_title("3. Analyse par regroupement")

# 2. 加载数据（示例，需替换为你自己的文件）
DATA_DIR_PROPRE = Path('baromètre/propres')

# 加载各表
@st.cache_data
def load_all():
    # 读取数据
    df = pyreadr.read_r(DATA_DIR_PROPRE / "donnees_completes_numérique.rds")[None]
    return df
df = load_all()
st.markdown("**Aperçu des données consolidées (année 2022)：**")
st.dataframe(df.head())

# 1. 只对数值型变量相关性
df_corr = df.select_dtypes(include=['number']).corr()

# 2. 画热力图
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_corr, cmap='vlag', annot=True, fmt=".2f", ax=ax,
            cbar=True, square=True, linewidths=.5, annot_kws={"size":8})

ax.set_title("Matrice de corrélation", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
st.pyplot(fig)

# 3. 预处理：去除缺失多的列/行，转换为数值型
df_num = df.select_dtypes(include=[np.number]).copy()
# 或手动选定你想分析的变量
# df_num = df[['décisions_avis_rendus','DMT_global', ...]]

# 1. KNN填补缺失
imputer = KNNImputer(n_neighbors=5)
df_filled = imputer.fit_transform(df_num)

# 2. Box-Cox变换 (PowerTransformer(method='box-cox') 只能用于所有值>0)
# 如果有<=0的数据可以用 Yeo-Johnson
try:
    pt = PowerTransformer(method='box-cox')
    df_boxcox = pt.fit_transform(df_filled)
except ValueError:
    pt = PowerTransformer(method='yeo-johnson')
    df_boxcox = pt.fit_transform(df_filled)

# 3. 标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_boxcox)

sub_title("3.1 PCA (après le prétraitement de données)")
# 5. 执行PCA
pca = PCA()
X_pca = pca.fit_transform(df_scaled)

explained_var = pca.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var)

# 6. 可视化PCA主成分贡献
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(range(1, len(explained_var)+1), explained_var*100, alpha=0.6, label="Variance expliquée %")
ax.plot(range(1, len(cum_explained_var)+1), cum_explained_var*100, marker='o', color='r', label="Cumul %")
ax.set_xlabel("Composante principale")
ax.set_ylabel("% Variance expliquée")
ax.set_title("Variance expliquée par composante (PCA)")
ax.legend()
st.pyplot(fig)

st.markdown(f"""
**Résultats principaux de la PCA :**  
- Les {np.argmax(cum_explained_var>=0.75)+1} premières dimensions expliquent {cum_explained_var[np.argmax(cum_explained_var>=0.75)]*100:.1f}% de la variance totale.
- Utilisation de ces dimensions pour la suite du regroupement (clustering).
""")

# 7. 展示主成分得分（可选，最多前2-3列）
pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
st.markdown("**Scores des observations sur les principales composantes:**")
st.dataframe(df_pca.iloc[:,:4].head())

sub_title("3.2 PAM")

# 4. PAM聚类 & silhouette分析
k_range = range(2, 11)  # 2~10类
silhouette_avgs = []
models = []

for k in k_range:
    kmedoids = KMedoids(n_clusters=k, random_state=2056, init='k-medoids++')
    labels = kmedoids.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    silhouette_avgs.append(score)
    models.append(kmedoids)

# 5. 可视化
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(k_range), silhouette_avgs, '-o', color=BLEU, alpha=0.7)
ax.set_xlabel('k (nombre de clusters)')
ax.set_ylabel('Largeur moyenne de silhouette')
ax.set_title("Choix du nombre optimal de clusters (PAM)")
ax.grid(False)
st.pyplot(fig)

# 6. 显示最佳K
k_range = range(2, 11)
best_k = k_range[np.argmax(silhouette_avgs)]
# best_k = 3
st.success(f"La meilleure partition conduit à **{best_k} groupes** (clusters), avec une largeur moyenne de silhouette = {max(silhouette_avgs):.2f}")
# 找到 k=3 在k_range的索引
best_model_idx = list(k_range).index(best_k)
# 7. 如需后续：直接用 models[np.argmax(silhouette_avgs)] 得到最佳模型
best_model = models[np.argmax(silhouette_avgs)]
labels = best_model.labels_
# 用这个模型与分组
# best_model = models[best_model_idx]
# labels = best_model.labels_


sub_title("3.3 Résultat du modèle")
# 把聚类结果加入df
df_num_with_cluster = df_num.copy()
df_num_with_cluster['cluster'] = labels
st.dataframe(df_num_with_cluster.head())

# Cluster建议转字符串/分类型
df_num_with_cluster['cluster'] = df_num_with_cluster['cluster'].astype(str)

# 选择变量（不包含cluster列）
variables = [col for col in df_num_with_cluster.columns if col != 'cluster']

# 数据拉成长表
df_long = df_num_with_cluster.melt(id_vars='cluster', value_vars=variables,
                                   var_name='Variable', value_name='Valeur')

# 使用美观的调色板（你可以用'set2', 'Set1', 'pastel', 'deep'等，下面是Set2）
palette = sns.color_palette("Set2")

# FacetGrid，每个变量独立y轴
g = sns.FacetGrid(df_long, col='Variable', col_wrap=4, sharey=False, height=3.2, aspect=1.0)
g.map_dataframe(sns.violinplot, x='cluster', y='Valeur', palette=palette, inner="quartile", cut=0)

g.set_titles('{col_name}')
g.set_axis_labels("Cluster", "Valeur")
for ax in g.axes.flatten():
    for label in ax.get_xticklabels():
        label.set_rotation(0)
    # 让每个y轴美观
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))

plt.tight_layout()
st.pyplot(g.fig)

sub_title("3.4 La carte")

# 加载donnees_completes.rds（有département列）
completes =pyreadr.read_r(DATA_DIR_PROPRE / "donnees_completes.rds")[None]
df_num_with_cluster['département'] = completes['département'].values

# 1. 读取地理数据
# 路径改成你自己的文件
shp_path = Path('contour départements')
gdf = gpd.read_file(shp_path / "france-departements-2024.shp")

# 2. 检查你的聚类结果DataFrame
# 假设你有df_num_with_cluster，包含列 'département' 和 'cluster'
df = df_num_with_cluster.copy()
# 字段名与shp一致，比如gdf['dep_current']，你就用gdf.merge(df, left_on="dep_current", right_on="département", ...)

st.write("gdf['dep_current'] unique:", sorted(gdf['dep_current'].unique()))
st.write("df['département'] unique:", sorted(df['département'].unique()))

# 常见做法：标准化部门代码长度
# df['département'] = df['département'].astype(str).str.zfill(2)
# gdf['dep_current'] = gdf['dep_current'].astype(str).str.zfill(2)

def clean_dep_current(dep):
    # 例如 dep="['01']"
    # 取出01, 2A, 976等
    match = re.findall(r"'([^']+)'", dep)
    if match:
        return match[0]
    else:
        # 保险起见
        return dep.replace('[','').replace(']','').replace("'", "").strip()

gdf['dep_current'] = gdf['dep_current'].apply(clean_dep_current)

# 3. 你的聚类结果 DataFrame，df_num_with_cluster
df = df_num_with_cluster.copy()  # 已含'département'和'cluster'

# 4. Corse（科西嘉）兼容处理
if '20' in df['département'].values:
    corse_row = df[df['département'] == '20']
    corse_2A = corse_row.copy()
    corse_2A['département'] = '2A'
    corse_2B = corse_row.copy()
    corse_2B['département'] = '2B'
    df = pd.concat([df, corse_2A, corse_2B], ignore_index=True)
    df = df[df['département'] != '20']

# 5. 再筛选一次，保证只有地图有的省份
df = df[df['département'].isin(gdf['dep_current'])]

# 3. 合并地理信息和分组标签
gdf = gdf.merge(df[['département','cluster']], left_on='dep_current', right_on='département', how='left')
st.write(gdf['cluster'].value_counts(dropna=False))

# 自定义配色：如有3个分组
cmap = ListedColormap(PALETTE[:df['cluster'].nunique()])

# dep_list_drom: 海外省的部门编号
dep_list_drom = ['971', '972', '973', '974', '976']

gdf_metropole = gdf[~gdf['dep_current'].isin(dep_list_drom)]
gdf_drom = gdf[gdf['dep_current'].isin(dep_list_drom)]

fig, ax = plt.subplots(figsize=(14,12))

# 主图: 本土
gdf_metropole.plot(
    column='cluster', categorical=True, legend=True, cmap=cmap,
    linewidth=0.25, edgecolor='black',
    ax=ax, missing_kwds={"color": "lightgrey", "hatch": "///", "label": "Aucune donnée"}
)
ax.set_title("Carte des clusters ( PAM)", fontsize=20)
ax.axis('off')

# 设定本土范围 (zoom in, 可微调)
ax.set_xlim(-5.5, 10)
ax.set_ylim(41, 51)

# ------- DROM插入 -------
drom_order = ["GUADELOUPE", "MARTINIQUE", "GUYANE", "LA REUNION", "MAYOTTE"]
drom_labels = {
    "971": "GUADELOUPE",
    "972": "MARTINIQUE",
    "973": "GUYANE",
    "974": "LA REUNION",
    "976": "MAYOTTE"
}
inset_pos = [
    [0.01, 0.85, 0.08, 0.08],  # GUADELOUPE
    [0.01, 0.75, 0.08, 0.08],  # MARTINIQUE
    [0.01, 0.65, 0.08, 0.08],  # GUYANE
    [0.01, 0.55, 0.08, 0.08],  # LA REUNION
    [0.01, 0.45, 0.08, 0.08],  # MAYOTTE
]
for i, dep in enumerate(dep_list_drom):
    if dep in gdf_drom['dep_current'].values:
        axin = fig.add_axes(inset_pos[i])
        gdf_drom[gdf_drom['dep_current']==dep].plot(
            column='cluster', categorical=True, legend=False, cmap=cmap,
            linewidth=0.25, edgecolor='black', ax=axin)
        axin.set_title(drom_labels[dep], fontsize=10, pad=0, loc='left')
        axin.axis('off')

plt.tight_layout()
st.pyplot(fig)

def generer_conclusion(df, cluster_col='cluster'):
    nb_clusters = df[cluster_col].nunique()
    counts = df[cluster_col].value_counts().sort_index()
    text = f"""
    #### Analyse des clusters

    Les départements ont été regroupés en **{nb_clusters} clusters** selon leurs caractéristiques (activité, délais, satisfaction, ressources humaines, etc.).

    - **Cluster 0** : {counts.iloc[0]} départements (en rouge sur la carte)
        - Ces départements présentent généralement **un volume d'activité plus élevé** (nombre de décisions rendues), **des effectifs plus importants**, des **délais moyens plus longs** et une **satisfaction plus faible** des usagers.
        - On les retrouve principalement dans les grandes régions urbaines ou surchargées, mais aussi dans certains départements d’Outre-mer.
        - Les indicateurs RH comme `étp_travaillés_total` sont élevés, la charge de travail par agent est souvent plus importante.

    - **Cluster 1** : {counts.iloc[1]} départements (en vert/bleu sur la carte)
        - Ces départements ont en moyenne **un volume d'activité plus modéré**, **des effectifs moindres**, mais **des délais plus courts** et une **satisfaction usagers plus élevée**.
        - Ils sont majoritairement situés dans des zones moins denses, rurales ou à taille plus humaine.
        - Les ressources sont plus faibles mais mieux adaptées, la charge par agent est moindre, l’écart ETP réel/théorique est aussi plus faible.

    - **Départements avec données manquantes :** {df[cluster_col].isna().sum()} (gris sur la carte).

    **Synthèse :**
    - Le cluster 0 concentre l’essentiel des situations de tension RH et de délais longs, avec des enjeux de satisfaction et d’organisation.
    - Le cluster 1, au contraire, présente une situation plus maîtrisée, avec une meilleure efficacité perçue par les usagers.

    > Ces résultats invitent à **renforcer l’accompagnement** des départements surchargés et à **valoriser les bonnes pratiques** des départements plus performants.

    """

    return text

# 在streamlit页面显示
st.markdown(generer_conclusion(df_num_with_cluster), unsafe_allow_html=True)
