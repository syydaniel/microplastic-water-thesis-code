# ============================================================  # 程序结构分隔线
# ① 环境适配与基础库导入
# ============================================================  # 程序结构分隔线

import os  # 导入操作系统接口模块，用于处理文件路径和系统级操作
import numpy as np  # 导入数值计算库，提供高性能矩阵运算和分位数统计功能
import pandas as pd  # 导入数据分析库，用于处理 Excel 表格及 DataFrame 数据清洗
import matplotlib  # 导入绘图库基础框架，用于配置底层渲染引擎
import matplotlib.backend_bases  # 导入绘图后端基类，用于兼容不同操作系统的窗口显示
import matplotlib.pyplot as plt  # 导入主绘图接口，提供类似 MATLAB 的便捷绘图指令
import matplotlib.gridspec as gridspec  # 导入高级布局管理器，支持精确控制非对称子图比例
from matplotlib.colors import LinearSegmentedColormap  # 导入颜色映射工具，用于创建学术渐变色板
from matplotlib.ticker import FormatStrFormatter  # 导入格式化工具，用于控制坐标轴数字的小数位精度
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 导入轴内嵌入工具，用于在图内精准放置色带
from matplotlib.lines import Line2D  # 导入几何线条类，用于手动绘制复杂的自定义图例标识
from scipy.signal import savgol_filter  # 导入信号滤波器，用于对趋势线进行平滑并提取二阶导数拐点

# 修复特定 Matplotlib 版本中的稳定性 Bug，防止底层 FigureManager 缺少状态属性导致程序崩溃
if not hasattr(matplotlib.backend_bases.FigureManagerBase, '_owns_mainloop'):
    matplotlib.backend_bases.FigureManagerBase._owns_mainloop = False  # 手动注入属性，确保窗口主循环正常运行

# 多环境自动适配逻辑：确保代码在 Jupyter 交互式环境、PyCharm 或 终端命令行下都能正常弹出绘图窗口
try:
    from IPython import get_ipython  # 尝试导入 IPython 核心组件
    if get_ipython() is None: matplotlib.use('TkAgg') # 如果不是在 Jupyter 笔记本中运行，则强制使用 TkAgg 图形界面后端
except Exception:
    matplotlib.use('TkAgg') # 若导入失败或环境异常，默认回退到 TkAgg 后端以保证图形窗口可见性

import lightgbm as lgb  # 【核心修改】：导入微软开发的 LightGBM 库，用于构建高效的梯度提升回归模型
import shap  # 导入模型解释库，基于博弈论计算特征对预测结果的影响力（SHAP 值）
from sklearn.model_selection import train_test_split, GridSearchCV  # 导入训练集分割工具和自动化网格搜索交叉验证工具

# ============================================================  # 程序结构分隔线
# ② 核心全局参数配置区（学术审美定制）
# ============================================================  # 程序结构分隔线

excel_file_path = r'test date.xlsx'  # 定义输入数据的文件路径（建议使用原始字符串 r 以避免转义字符错误）
target_column = 'MB'  # 指定数据集中需要进行预测的目标变量列名

# 定制符合顶级学术期刊（如 Nature 风格）的配色方案：深蓝代表低值，浅灰代表中性，深红代表高值
main_cmap_colors = ["#7B6C9F", "#e0e0e0", "#3AB5B3"] 
custom_cmap = LinearSegmentedColormap.from_list('val_cmap', main_cmap_colors) # 将颜色列表转换为平滑的线性映射对象

# 全局字体规范设置：确保生成的图表文字在论文排版中具有高度可读性
plt.rcParams['font.family'] = 'serif' # 指定字体族为衬线体
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun'] # 优先使用 Times New Roman，中文字符自动映射为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴中负号显示为乱码方块的问题

# 学术可视化审美参数字典：集中控制全局视觉比例，确保图表元素（线宽、字号）的统一性
aesthetic_params = {
    'ax_label_size': 22,      # 设置坐标轴标题（如：特征名称）的字号大小
    'tick_label_size': 20,    # 设置坐标轴刻度数字（如：0.1, 0.2）的字号大小
    'legend_size': 20,        # 设置图例内部文字的字号大小
    'line_width': 1.5,        # 设置 PDP (部分依赖图) 趋势线的宽度
    'marker_size': 35,        # 设置 SHAP 散点图中的数据点大小
    'edge_width': 0.6,        # 设置散点的白色外边缘粗细，增强高密度区域的辨识度
    'spine_width': 1.5        # 设置坐标轴四周边框的线条粗细
}

# 局部样式细节配置：定义 PDP 趋势曲线及其置信区间阴影的视觉属性
pdp_style = {'line_color': "#000000", 'ci_color': '#333333', 'ci_alpha': 0.1} 
# 底部频数直方图配置：使用学术淡蓝色，配合白色描边以增加层次感
hist_style = {'color': "#7B6C9F", 'edgecolor': 'white', 'linewidth': 0.8, 'alpha': 0.6} 

# ============================================================  # 程序结构分隔线
# ③ 算法辅助功能库（逻辑实现）
# ============================================================  # 程序结构分隔线

def apply_academic_style(ax, close_top=True):
    """应用学术级封闭式坐标轴样式：设置边框、刻度方向及图层深度"""
    for spine in ax.spines.values(): # 遍历上下左右四个边框线
        spine.set_linewidth(aesthetic_params['spine_width']) # 统一设置边框线宽
        spine.set_visible(True) # 确保所有边框线均显示
    if not close_top: ax.spines['top'].set_visible(False) # 如果不需要顶部闭合，则隐藏顶部线
    # 设置刻度线向外凸出（学术规范），设置刻度线长度、宽度及文字大小
    ax.tick_params(direction='out', length=6, width=1.2, labelsize=aesthetic_params['tick_label_size'])
    ax.set_axisbelow(True) # 将坐标轴网格和背景线置于数据层之下，防止遮挡数据点

def auto_smart_legend(ax, handles, labels, x_data, y_data):
    """智能避让算法：通过计算数据在四个象限的密度分布，自动将图例放置在最空旷的区域"""
    mask = ~np.isnan(x_data) & ~np.isnan(y_data) # 过滤掉数据中的空值，避免计算报错
    x, y = np.array(x_data)[mask], np.array(y_data)[mask] # 转换为 NumPy 数组并应用掩码
    if len(x) == 0: # 如果没有有效数据
        ax.legend(handles=handles, labels=labels, loc='best', frameon=False) # 使用系统默认位置
        return
    x_mid, y_mid = (np.max(x) + np.min(x)) / 2, (np.max(y) + np.min(y)) / 2 # 计算当前坐标轴显示区域的中点
    # 统计落在右上、左上、左下、右下四个象限的点数
    q1 = np.sum((x > x_mid) & (y > y_mid)) # 第一象限
    q2 = np.sum((x <= x_mid) & (y > y_mid)) # 第二象限
    q3 = np.sum((x <= x_mid) & (y <= y_mid)) # 第三象限
    q4 = np.sum((x > x_mid) & (y <= y_mid)) # 第四象限
    counts = {'upper right': q1, 'upper left': q2, 'lower left': q3, 'lower right': q4} # 密度字典
    best_loc = min(counts, key=counts.get) # 挑选点数最少（最空旷）的象限
    ax.legend(handles=handles, labels=labels, loc=best_loc, fontsize=aesthetic_params['legend_size'], frameon=False) # 绘制无边框图例

def find_knee_point(x_data, y_data):
    """数学敏感度分析：利用二阶导数寻找 SHAP 贡献发生剧烈变化的拐点，即特征响应的“敏感阈值”"""
    if len(x_data) < 10: return np.median(x_data) # 样本量太少时无法计算导数，返回中位数作为默认值
    sorted_idx = np.argsort(x_data) # 获取 X 轴数据的排序索引
    x_s, y_s = x_data.values[sorted_idx], y_data[sorted_idx] # 对数据按特征值大小进行排序
    try:
        y_smooth = savgol_filter(y_s, 11, 2, deriv=2) # 使用滤波器平滑数据并计算二阶导数（曲率）
        knee_idx = np.argmax(np.abs(y_smooth)) # 寻找曲率绝对值最大的点，即为趋势发生质变的位置
        return x_s[knee_idx] # 返回该拐点对应的特征取值
    except: return np.median(x_data) # 如果数学计算异常，回退到中位数

def compute_pdp_with_ci(model, X_ref, feature, grid_points=30, n_boot=50):
    """Bootstrap 重采样计算：生成部分依赖图 (PDP) 的均值线及其 95% 置信区间阴影"""
    x = X_ref[feature].dropna().values # 提取目标特征列并去除空值
    if len(np.unique(x)) < 5: return None, None, None # 若特征取值过于稀疏（如二分类），则不适合绘制连续 PDP
    x_min, x_max = np.percentile(x, [1, 99]) # 剔除极值，选取 1% 到 99% 的范围作为绘图主区间
    grid = np.linspace(x_min, x_max, grid_points) # 在该区间内生成均匀分布的计算网格点
    boot_curves = [] # 用于存储每一轮重采样的预测结果
    X_copy = X_ref.copy() # 建立参考数据集的深拷贝
    for _ in range(n_boot): # 执行 Bootstrap 循环
        resample_idx = np.random.choice(len(X_copy), len(X_copy), replace=True) # 有放回地随机抽取样本
        X_res = X_copy.iloc[resample_idx] # 构建重采样后的数据集
        # 对网格中的每一个值，强制修改全样本的该特征，并计算模型预测的平均值
        curve = [model.predict(X_res.assign(**{feature: v})).mean() for v in grid]
        boot_curves.append(curve) # 记录该曲线
    # 返回网格点、所有曲线的均值、以及 2.5% 到 97.5% 之间的置信空间范围
    return grid, np.mean(boot_curves, axis=0), (np.quantile(boot_curves, 0.025, axis=0), np.quantile(boot_curves, 0.975, axis=0))

# ============================================================  # 程序结构分隔线
# ④ 数据载入、建模与 SHAP 计算
# ============================================================  # 程序结构分隔线

print("--> 正在载入数据并构建 LightGBM 解释性模型...") # 在控制台打印任务进度
df = pd.read_excel(excel_file_path) # 从指定的路径读取 Excel 数据集
X = df.drop(columns=[target_column]) # 特征矩阵：保留除目标列以外的所有数据
y = df[target_column] # 目标向量：仅提取需要预测的变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 按照 7:3 比例切分训练集和测试集

# 【核心模型替换点】：使用 LightGBM 回归器进行参数寻优
# verbosity=-1 设置为静默模式，避免输出冗余的训练日志
gs = GridSearchCV(lgb.LGBMRegressor(verbosity=-1, random_state=42), 
                  {
                      'n_estimators': [500],      # 弱学习器的迭代次数（对应 CatBoost 的 iterations）
                      'max_depth': [6],           # 决策树的最大深度，控制模型复杂度
                      'learning_rate': [0.05],    # 学习步长，决定模型收敛速度
                      'num_leaves': [31]          # LightGBM 核心参数：叶子节点数
                  }, cv=5) # 采用 5 折交叉验证寻找最优超参数组合

gs.fit(X_train, y_train) # 在训练集上执行拟合任务
model = gs.best_estimator_ # 获取经过交叉验证筛选出的最佳模型实例

explainer = shap.TreeExplainer(model) # 创建专门针对树集成模型的 SHAP 解释器
shap_obj = explainer(X_test) # 计算测试集上每个样本、每个特征的 SHAP 贡献值
shap_values = shap_obj.values # 获取 SHAP 值的矩阵，用于后续所有可视化分析

# ============================================================  # 程序结构分隔线
# ⑤ 核心绘图流程：Publication-ready 图表生成
# ============================================================  # 程序结构分隔线

fig = plt.figure(figsize=(32, 28)) # 创建一个极高分辨率的大尺寸画布，满足高质量排版需求
# 建立顶层网格：3行4列。前两列用于展示全局汇总图，后两列用于展示具体特征的详情分析
gs_main = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[0.7, 0.7, 2.3, 2.3], 
                            left=0.04, right=0.96, wspace=0.7, hspace=0.35)

# -------------------- ⑤.1 左侧：Global Summary 蜂窝图 --------------------
ax_summary = fig.add_subplot(gs_main[:, :2]) # 合并左侧两列的全部行，用于绘制全局重要性汇总图
mean_shaps = np.abs(shap_values).mean(axis=0) # 计算每个特征 SHAP 值的平均绝对值，作为特征全局重要性排序依据
feat_ranking = pd.DataFrame({'f': X_test.columns, 'imp': mean_shaps}).sort_values('imp', ascending=True) # 升序排列特征

ax_summary.set_yticks(range(len(feat_ranking))) # 设定 Y 轴刻度位置
ax_summary.set_yticklabels(feat_ranking['f'], fontsize=aesthetic_params['tick_label_size']) # 将 Y 轴标签设置为特征名称

ax_top_bar = ax_summary.twiny() # 创建顶部孪生轴，用于在背景绘制条形图以辅助量化重要性
ax_top_bar.barh(range(len(feat_ranking)), feat_ranking['imp'], color="#cad0ea", alpha=0.3, height=0.6) # 绘制浅色半透明背景条
ax_top_bar.set_xlabel("Mean |SHAP| (Importance)", fontsize=aesthetic_params['ax_label_size'], labelpad=12) # 设定顶部轴标题
apply_academic_style(ax_top_bar) # 应用学术排版样式

# 遍历每一个特征，绘制对应的蜂窝状散点分布
for i, f_name in enumerate(feat_ranking['f']):
    idx = X_test.columns.get_loc(f_name) # 获取特征在矩阵中的列索引
    sv, fv = shap_values[:, idx], X_test.iloc[:, idx] # 提取对应的 SHAP 值向量和原始特征值向量
    # 绘制散点：Y 坐标引入正态分布随机抖动 (Jitter)，避免数据点完全重叠。颜色由原始特征值高低映射。
    ax_summary.scatter(sv, i + np.random.normal(0, 0.08, len(sv)), c=fv, cmap=custom_cmap, 
                       s=aesthetic_params['marker_size'], edgecolors='white', 
                       linewidths=aesthetic_params['edge_width'], alpha=0.8, zorder=3)

ax_summary.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=aesthetic_params['ax_label_size']) # 设置底部轴主标题
apply_academic_style(ax_summary) # 应用学术排版样式

# 在左侧汇总图右侧插入一个微型色带轴，用于说明特征值的高低颜色含义
cax_sum = inset_axes(ax_summary, width="3%", height="100%", loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax_summary.transAxes, borderpad=0)
sm_sum = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(0, 1)) # 定义色带映射比例
cb_sum = fig.colorbar(sm_sum, cax=cax_sum) # 渲染色带
cb_sum.set_label('Feature Value (Normalized)', fontsize=aesthetic_params['ax_label_size'], labelpad=10) # 色带标签
cb_sum.ax.tick_params(labelsize=aesthetic_params['tick_label_size']) # 色带刻度字号
cb_sum.outline.set_visible(False) # 移除色带外框线，保持视觉简洁

# -------------------- ⑤.2 右侧：PDP & SHAP 详情图 --------------------
top_6_list = feat_ranking['f'].tail(6).iloc[::-1].tolist() # 挑选重要性排名前 6 的核心特征进行深入剖析
detail_grid = gs_main[:, 2:].subgridspec(3, 2, wspace=0.4, hspace=0.15) # 在右侧两列内建立一个 3x2 的细分子网格

for idx, feature in enumerate(top_6_list): # 循环绘制每一个重要特征的子图
    r, c = idx // 2, idx % 2 # 计算当前特征应该放在 3x2 网格的哪一行、哪一列
    # 在每个子单元格内再次划分：上方为主图（高度 7），下方为分布直方图（高度 1），右侧为局部色带
    inner_gs = detail_grid[r, c].subgridspec(2, 2, height_ratios=[7, 1], width_ratios=[30, 1.2], hspace=0.08, wspace=0)
    
    ax_pdp = fig.add_subplot(inner_gs[0, 0])      # 创建承载 PDP 均值趋势线的轴
    ax_hist = fig.add_subplot(inner_gs[1, 0], sharex=ax_pdp) # 创建承载频数直方图的轴，并与主图共享 X 轴刻度
    ax_cb_sub = fig.add_subplot(inner_gs[0, 1])   # 创建用于放置子图局部色带的微型轴
    ax_shap = ax_pdp.twinx()                     # 创建主图轴的孪生 Y 轴，用于在同一位置显示右侧的 SHAP 散点值

    # 【图层控制技术】：解决多轴重叠导致的遮挡问题
    ax_pdp.set_zorder(10) # 强制将 PDP 均值线所在轴置于最顶层
    ax_pdp.patch.set_visible(False) # 将主图轴背景设为透明，否则底层的孪生轴（散点图）会被遮挡无法显示
    ax_shap.set_zorder(1) # 将绘制散点的轴降到底层，作为背景数据分布展示

    x_vec = X_test[feature] # 提取当前特征的观测值
    f_idx = X_test.columns.get_loc(feature) # 获取对应的列索引
    y_vec_shap = shap_values[:, f_idx] # 获取对应的 SHAP 向量
    v_abs = max(abs(np.nanpercentile(y_vec_shap, 1)), abs(np.nanpercentile(y_vec_shap, 99))) # 计算 SHAP 对称色彩范围
    
    # 绘制孪生轴中的散点图：展示每个样本的具体贡献度，颜色随 SHAP 值大小变化
    sc_plot = ax_shap.scatter(x_vec, y_vec_shap, c=y_vec_shap, cmap=custom_cmap, 
                             norm=plt.Normalize(-v_abs, v_abs), 
                             s=aesthetic_params['marker_size'], edgecolors='white', 
                             linewidths=aesthetic_params['edge_width'], alpha=0.7)
    
    # 执行 Bootstrap 计算得到 PDP 均值和置信区间
    g_x, g_y, g_ci = compute_pdp_with_ci(model, X_test, feature)
    if g_x is not None:
        # 在主轴绘制黑色的模型响应趋势线（PDP 均值线）
        ax_pdp.plot(g_x, g_y, color=pdp_style['line_color'], lw=aesthetic_params['line_width'], zorder=5)
        # 填充代表不确定性的 95% 置信区间灰色半透明阴影
        ax_pdp.fill_between(g_x, g_ci[0], g_ci[1], color=pdp_style['ci_color'], alpha=pdp_style['ci_alpha'], lw=0, zorder=4)

    # 在底部副轴绘制特征值的分布直方图，帮助判断哪些区域的数据更具代表性
    ax_hist.hist(x_vec.dropna(), bins=35, **hist_style)
    ax_hist.set_xlabel(feature, fontsize=aesthetic_params['ax_label_size']) # X 轴标签即为特征名
    ax_hist.set_ylabel("Count", fontsize=aesthetic_params['ax_label_size']) # Y 轴为样本频数
    apply_academic_style(ax_hist) # 应用学术样式

    ax_pdp.set_ylabel("PDP Trend", fontsize=aesthetic_params['ax_label_size']) # 左侧 Y 轴标签
    ax_pdp.tick_params(labelbottom=False) # 隐藏主图底部刻度数字，避免与下方直方图的数字重叠
    ax_shap.set_yticks([]) # 隐藏右侧孪生轴的刻度数字，通过紧邻的色带获取数值参考即可
    apply_academic_style(ax_pdp, close_top=True) # 应用学术样式并闭合顶部边框线
    ax_shap.tick_params(direction='out', length=6, width=1.2) # 设置右侧刻度样式
    apply_academic_style(ax_shap, close_top=True) # 再次应用样式确保边框完整性

    # 针对当前特征图，绘制其专属的 SHAP 数值色带
    cb_sub = fig.colorbar(sc_plot, cax=ax_cb_sub)
    cb_sub.outline.set_visible(False) # 隐藏色带框线
    cb_sub.ax.tick_params(labelsize=aesthetic_params['tick_label_size'], direction='out', length=4) # 设置刻度字号
    cb_sub.set_label("SHAP Value", fontsize=aesthetic_params['ax_label_size'], labelpad=5) # 色带侧面标签
    cb_sub.formatter = FormatStrFormatter('%.1f') # 限制色带刻度为一位小数
    cb_sub.update_ticks() # 强制渲染更新刻度显示

    # 计算统计辅助线：包括该特征数据的中位数线，以及通过二阶导数识别出的敏感拐点（Knee Point）
    v_median, v_threshold = x_vec.median(), find_knee_point(x_vec, y_vec_shap)
    l1 = ax_pdp.axvline(v_median, color='black', ls='--', lw=1.2, alpha=0.5, zorder=2) # 黑色虚线表示中位数
    l2 = ax_pdp.axvline(v_threshold, color='red', ls=':', lw=1.5, alpha=0.6, zorder=2) # 红色点线表示敏感阈值
    
    # 构造自定义图例元素列表
    handles = [Line2D([0], [0], color='black', lw=2), l1, l2]
    labels = ['PDP Trend', f'Median: {v_median:.1f}', f'Threshold: {v_threshold:.1f}']
    # 调用智能算法，将图例放置在图像中最空旷的位置，防止遮挡关键趋势
    auto_smart_legend(ax_pdp, handles, labels, x_vec.values, y_vec_shap)

# ============================================================  # 程序结构分隔线
# ⑥ 图像保存与渲染
# ============================================================  # 程序结构分隔线

output_file = f"Result_LGBM_{target_column}.png" # 定义输出的高清图片文件名，加入 LGBM 标识以示区分
plt.savefig(output_file, dpi=300, bbox_inches='tight') # 以 300DPI（印刷级标准）保存图像，自动裁剪边缘留白
print(f"--> [程序运行成功] 高清图像已保存至: {output_file}") # 在控制台输出确认信息
plt.show() # 激活图形界面，在屏幕上直接弹出显示绘制完成的图表