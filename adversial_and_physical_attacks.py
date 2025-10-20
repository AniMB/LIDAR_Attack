#%%
import random
import os
import csv
import ast
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


from scipy.stats import *
from astropy.stats import knuth_bin_width
from sklearn.neighbors import *
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
from pathlib import Path
#%% md
# | filename       | type(simulated/real) | class(benign/malignant) | voxel                    |
# |----------------|----------------------|-------------------------|--------------------------|
# | real/00000.bin | real                 | benign                  | "(00 000 000 000 00 00)" |
#%%

class RowEntry:
    def __init__(self, filename):
        self.filename: str=filename
        self.type:str=self.filename.split('/')[0].split('_')[0]
        self.class_:int=0
        self.voxel:str=""


    def to_csv(self):
        return [self.filename, self.type, self.class_, self.voxel]
#%%
file=open("filetracker.csv","+w",newline="")
writer=csv.writer(file)
writer.writerow(["filename","type(sim/real)","class(Benign=0/Malignant=1)","Voxel"])
def add_entries(dirname):
    for roots, dirs, files in os.walk(dirname):
        for filenames in files:
            entry_ = RowEntry(f"{dirname}/{filenames}")
            writer.writerow(entry_.to_csv())
add_entries("real_world_training")
# add_entries("simulated_training")
file.close()


# store information about every file as to its name and whether or not is benign or malignant



#%% md
# ### Poisoning the data
#%%
df=pd.read_csv("filetracker.csv", header=0)
poison_percent=0.05
idx:list=[random.randint(0,len(df)-1) for i in range(int(poison_percent*len(df))) ]
df.loc[idx,"class(Benign=0/Malignant=1)"]=1
df.to_csv("filetracker_poisoned.csv",index=False)
#%% md
# ### Voxelize the data into around 20 features after converting into readable information. Write this text back into the csv file tha will store the struct.
#%%
# features=[x_mean, x_max, x_min, x_std, y_mean, y_std, y_min, y_max, z_mean, z_min, z_max, z_std, r_max, r_mean, r_std, r_min, kurtosis_r, skewness_in_r, percentage_outlier_r,Peak_bin_ratio_r entropy_of_r,range_x, range_y, range_z, range_r,curvature_reflectance_region_ratio]
class Voxelize:
    def __init__(self, pointcloud: np.ndarray):
        self.__pointcloud = pointcloud

    def __mean(self,col)-> float:
        return self.__pointcloud[:,col].mean()

    def __std(self,col)-> float:
        return self.__pointcloud[:,col].std()

    def __min(self,col)-> float:
        return self.__pointcloud[:,col].min()

    def __max(self,col)-> float:
        return self.__pointcloud[:,col].max()

    def __range(self,col)-> float:
        return self.__pointcloud[:,col].max()-self.__pointcloud[:,col].min()

    def __entropy_r(self)-> float:
        data=self.__pointcloud[:,3]
        width, bin_edges = knuth_bin_width(data, return_bins=True)
        hist,_=np.histogram(data, bins=bin_edges)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        return entropy(probs, base=2)

    def __kurtosis_r(self)-> float:
        return kurtosis(self.__pointcloud[:,3])

    def __skewness_r(self)-> float:
        return skew(self.__pointcloud[:,3],axis=0,bias=False)

    def __percentage_outliers_r(self)-> float:
        data = self.__pointcloud[:,3].reshape(-1, 1)
        n_neighbors = int(min(max(0.05 * len(data), 20), 100))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        preds = lof.fit_predict(data)
        return np.sum(preds == -1) / len(data)

    def __peak_bin_ratio_r(self)-> float:
        data=self.__pointcloud[:,3]
        width, bin_edges = knuth_bin_width(data, return_bins=True)
        hist,_=np.histogram(data, bins=bin_edges)
        return np.max(hist)/np.sum(hist)

    def __curvature_reflectance_region_ratio(self)-> float:
        points=self.__pointcloud[:,:3]
        nbr = NearestNeighbors(n_neighbors=122).fit(points)
        _, indices = nbr.kneighbors(points)
        curvatures = []
        for i in range(len(points)):
            neighbors = points[indices[i]]
            centroid = np.mean(neighbors, axis=0)
            cov = np.cov((neighbors - centroid).T)

            eigvals = np.linalg.eigvalsh(cov)  # returns sorted
            if np.sum(eigvals) > 0:
                curvature = eigvals[0] / np.sum(eigvals) # citation exists
            else:
                curvature = 0
            curvatures.append(curvature)

        curvatures=np.array(curvatures)

        reflectance = self.__pointcloud[:,3].reshape(-1, 1)
        reflectance += np.random.normal(0, 1e-5, size=reflectance.shape)

        lof = LocalOutlierFactor(n_neighbors=122, contamination='auto')
        preds = lof.fit_predict(reflectance)
        high_r=np.array(preds==-1)

        high_r_curv=curvatures[high_r]

        return np.mean(high_r_curv)/np.mean(curvatures)


    def voxel(self)->tuple:
        return tuple([self.__mean(0), self.__min(0), self.__max(0), self.__range(0), self.__std(0),
                      self.__mean(1), self.__min(1), self.__max(1), self.__range(1), self.__std(1),
                      self.__mean(2), self.__min(2), self.__max(2), self.__range(2), self.__std(2),
                      self.__mean(3), self.__min(3), self.__max(3), self.__range(3), self.__std(3),
                      self.__entropy_r(), self.__kurtosis_r(), self.__skewness_r(),self.__peak_bin_ratio_r(),
                      self.__curvature_reflectance_region_ratio()
                      ])
#%% md
# ### Adding Simulations
#%%
class ROISaturationInjector:
    """
    Implements the 'ROI Saturation Injection with Smooth Intensities' algorithm.
    """

    def __init__(self, point_cloud: np.ndarray, alpha: float = 0.2):
        """
        @brief Initializes the injector with a point cloud and a smoothness factor.

        The point cloud is expected to have shape (N, 4) with columns for x, y, z, and intensity.

        Args:
            point_cloud (np.ndarray): The input point cloud, with shape (N, 4).
                                      The fourth column is expected to be intensity values [0, 1].
            alpha (float): The smoothness factor for intensity noise [0.1, 0.3].
        """
        if point_cloud.shape[1] != 4:
            raise ValueError("Input point cloud must have 4 columns (x, y, z, r).")

        if not (0.1 <= alpha <= 0.3):
            print(f"Warning: alpha={alpha} is outside the recommended [0.1, 0.3] range.")

        self.points = point_cloud[:, :3]
        self.intensities = point_cloud[:, 3]
        self.alpha = alpha
        self.epsilon = 1e-8

        # Internal properties to be calculated upon injection
        self.injected_points = None
        self.injected_intensities = None

    def inject(self, roi_bounds: tuple, n_injection: int) ->  np.ndarray:
        """
        @brief Applies the saturation injection algorithm to the point cloud.
        The point cloud and alpha are taken from the class instance.

        Args:
            roi_bounds (tuple): A tuple defining the ROI as a bounding box:
                                (xmin, ymin, zmin, xmax, ymax, zmax).
            n_injection (int): The number of new points to inject.

        Returns:
            A tuple containing:
                - np.ndarray: The augmented point cloud, shape (N + n_injection, 3)

        """
        print("--- Starting ROI Saturation Injection ---")

        # === Step 1: Select compact ROI data ===
        xmin, ymin, zmin, xmax, ymax, zmax = roi_bounds
        in_roi_mask = (
            (self.points[:, 0] >= xmin) & (self.points[:, 0] <= xmax) &
            (self.points[:, 1] >= ymin) & (self.points[:, 1] <= ymax) &
            (self.points[:, 2] >= zmin) & (self.points[:, 2] <= zmax)
        )
        positive_intensity_mask = self.intensities > 0
        seed_points_mask = in_roi_mask & positive_intensity_mask

        roi_seed_points = self.points[seed_points_mask]
        roi_seed_intensities = self.intensities[seed_points_mask]

        if roi_seed_points.shape[0] < 2:
            print("Warning: Fewer than 2 valid seed points found in ROI. "
                  "Cannot fit model. Returning the original cloud.")
            return self.points
        print(f"Found {roi_seed_points.shape[0]} seed points in ROI.")

        # === Step 2: Fit intensity constant (A_hat) ===
        ranges = np.linalg.norm(roi_seed_points, axis=1)
        x_i = 1 / (ranges**2 + self.epsilon)
        y_i = roi_seed_intensities

        numerator = np.sum(x_i * y_i)
        denominator = np.sum(x_i**2)

        if abs(denominator) < self.epsilon:
             print("Warning: Denominator for A_hat is near zero. "
                   "Cannot fit model. Returning original cloud.")
             return self.points
        a_hat = numerator / denominator
        print(f"Fitted intensity constant A_hat = {a_hat:.4f}")

        # === Step 3: Sample new positions in ROI ===
        new_points_x = np.random.uniform(xmin, xmax, n_injection)
        new_points_y = np.random.uniform(ymin, ymax, n_injection)
        new_points_z = np.random.uniform(zmin, zmax, n_injection)
        injected_points = np.vstack([new_points_x, new_points_y, new_points_z]).T
        print(f"Sampled {n_injection} new positions in the ROI.")

        # === Step 4: Assign smooth intensities ===
        injected_ranges = np.linalg.norm(injected_points, axis=1)

        ideal_intensities = a_hat / (injected_ranges**2 + self.epsilon)

        variance = self.alpha * np.abs(ideal_intensities)
        std_dev = np.sqrt(variance)
        noise = np.random.normal(loc=0.0, scale=std_dev)

        injected_intensities = np.clip(ideal_intensities + noise, 0, 1)
        print("Assigned smooth intensities to new points.")

        # === Step 5: Merge & Preprocess ===
        augmented_point_cloud = np.vstack([self.points, injected_points])

        print("Merged original and injected points.")
        print("--- Injection Complete ---")

        return augmented_point_cloud

#%% md
# ### Seperator
#%%
def save_kitti_bin(points_xyz_i: np.ndarray, out_path: str | Path):
    """
    Save point cloud to KITTI-style .bin (x, y, z, intensity) as float32.

    points_xyz_i: (N, 3) or (N, 4) numpy array
                  columns: x, y, z [, intensity]
    out_path:     output filename ending with .bin
    """
    pts = np.asarray(points_xyz_i)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError("points_xyz_i must be (N,3) or (N,4)")

    # If no intensity, add zeros
    if pts.shape[1] == 3:
        zeros = np.zeros((pts.shape[0], 1), dtype=np.float32)
        pts = np.hstack([pts.astype(np.float32, copy=False), zeros])
    else:
        pts = pts.astype(np.float32, copy=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts.tofile(out_path)   # little-endian float32 by default

df=pd.read_csv("filetracker_poisoned.csv", header=0)
# --- Partition and Injection Section ---
df = pd.read_csv("filetracker_poisoned.csv", header=0)

class RandomPartitioner:
    def __init__(self, df, seed=None):
        self.df = df
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def partition(self, p=0.5):
        labels = np.random.choice([0, 1], size=len(self.df), p=[1 - p, p])
        df_out = self.df.copy()
        df_out["class(Benign=0/Malignant=1)"] = labels
        return df_out


partitioner = RandomPartitioner(df, time.time())
df_labeled = partitioner.partition()

for idx, row in df_labeled.iterrows():
    if row["class(Benign=0/Malignant=1)"] == 1:
        point_cloud = np.fromfile(row["filename"], dtype=np.float32).reshape(-1, 4)
        injector = ROISaturationInjector(point_cloud)
        x, y, z, r = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3]
        x_max, x_min, y_max, y_min, z_max, z_min = x.max(), x.min(), y.max(), y.min(), z.max(), z.min()

        random.seed(time.time())
        random_x = random.uniform(x_min, x_max)
        random_y = random.uniform(y_min, y_max)
        random_z = random.uniform(z_min, z_max)
        random_points = random.randint(30, 40)

        injected_points = injector.inject(
            (random_x, random_y, random_z, random_x + 10, random_y + 10, random_z + 10),
            random_points,
        )

        file_name = f"injected/{os.path.basename(row['filename'])}"
        save_kitti_bin(injected_points, file_name)
        df_labeled.loc[idx, "filename"] = file_name

df_labeled.to_csv("filetracker_poisoned_injected.csv", index=False)




#%% md
# ### Adding Voxels
# 
#%%
df=pd.read_csv("filetracker_poisoned_injected.csv", header=0)
df['voxel'] = [
    str(tuple(Voxelize(np.fromfile(fname, dtype=np.float32).reshape(-1, 4)).voxel()))
    for fname in df['filename']
]
df.to_csv("filetracker_poisoned_voxelized.csv",index=False)


#%% md
# ### Preprocessing
#%%
df = pd.read_csv("filetracker_poisoned_voxelized.csv", header=0, dtype={"Voxel": str})

y_data=df['class(Benign=0/Malignant=1)'].iloc[:9286].to_numpy()


# Extract first 9286 entries from the 'voxel' column
voxel_strings = df["voxel"].iloc[:9286]


parsed_voxels = []
errors = []

for idx, val in enumerate(voxel_strings):
    if isinstance(val, str) and val.startswith("(") and val.endswith(")"):
        try:
            parsed_voxels.append(ast.literal_eval(val))
        except Exception as e:
            errors.append((idx, val, str(e)))
    else:
        errors.append((idx, val, "Invalid format"))


x_data = np.array(parsed_voxels)
print(errors)







# # adding noise in simulated.
# noise_level=1e-5
# x_data_noisy=x_data.copy()
# indices=df.index[df['type(sim/real)']=='simulated']
# x_data_noisy[indices] += np.random.normal(0, noise_level,x_data_noisy[indices].shape )










#%% md
# ### Conducting Detailed Ananlysis using differnt ML classifiers.
#%%


# Conducting PCA reducing to 10 dimensions and then cross validating, proceeding in direction of generating f-scores for models using SVM, Random Forest, ADAboost, kNN, aNN, Decision Tree, Gradient Boosting, Voting Classifier.

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.25, random_state=42, stratify=None
)


rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
svm = SVC(
    kernel='rbf',
    C=0.1,
    gamma='scale',
    probability=True,
    random_state=42
)
knn = KNeighborsClassifier(
    n_neighbors=9
)
dt = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=0.1,
    random_state=42
)
gb = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)


voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf), ('svm', svm), ('knn', knn),
        ('dt', dt), ('ada', ada), ('gb', gb)
    ],
    voting='soft'
)

classifiers = {
    "RandomForest": rf,
    "SVM": svm,
    "kNN": knn,
    "DecisionTree": dt,
    "AdaBoost": ada,
    "GradientBoosting": gb,
    "VotingClassifier": voting_clf
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_score = 0
dim_keep=10
for name, clf in classifiers.items():
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=dim_keep)),
            ('clf', clf)
        ])
    try:

        scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
        mean_score = np.mean(scores)
    except:

        scores = cross_val_score(clf, X_train, y_train, cv=5)
        mean_score = np.mean(scores)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    if test_acc > best_score:
        best_score = test_acc
        best_model = clf

    print(f"▶ {name} ◀")
    print(f"  - CV Accuracy : {mean_score:.4f}")
    print(f"  - Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\n✔ Best Model:", best_model)
y_pred_best = best_model.predict(X_test)
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))


#%% md
# ### Graphing the outputs using a learning curve
#%%
colors = plt.cm.get_cmap('tab10', len(classifiers))
train_sizes = np.linspace(0.1, 1.0, 10)
plt.figure(figsize=(12, 7))

for idx, (name, clf) in enumerate(classifiers.items()):
    train_sizes_abs, train_scores, val_scores = learning_curve(
        clf, X_train, y_train, cv=cv, train_sizes=train_sizes, scoring="accuracy"
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    color = colors(idx)
    plt.plot(train_sizes_abs, train_mean, marker="o", linestyle="-", linewidth=2, color=color,
             label=f"{name} - Training")
    plt.plot(train_sizes_abs, val_mean, marker="o", linestyle="--", linewidth=1.5, color=color,
             label=f"{name} - Validation")

plt.title(("Learning Curves of Denial of Service Dataset"), fontsize=16, fontweight='bold')
plt.xlabel("Training Size", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.legend(loc="upper left", fontsize=11, frameon=True, ncol=2, bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()
#%% md
# # POint cloud processing for reference
#%%
point_cloud = np.fromfile('00174.bin', dtype=np.float32).reshape(-1, 4)
print(type(point_cloud))
print(point_cloud.shape)
np.savetxt("converted.txt", point_cloud, fmt="%.6f", delimiter=",")
print(len(point_cloud))
#%% md
# ### Visulaisation in matplotlib
#%%
x, y, z, r = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3]
r_norm = (r - r.min()) / (r.max() - r.min())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=r_norm, cmap='viridis', s=0.5)
plt.show()
#%% md
# ## Visualisation in open3d
#%%
# Create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
pcd.colors = o3d.utility.Vector3dVector(
    np.stack([r_norm, np.zeros_like(r_norm), np.zeros_like(r_norm)], axis=1)
)

# Optional: voxel grid (commented out if not needed)
voxel_size = 0.2
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# Initialize visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='LiDAR Point Cloud', width=1280, height=720)
vis.add_geometry(pcd)

# Render settings
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
opt.point_size = 2.0
opt.show_coordinate_frame = False

# Center the camera on the point cloud
ctr = vis.get_view_control()
ctr.set_lookat(pcd.get_center())
ctr.set_front([0.0, 0.0, -1.0])
ctr.set_up([0.0, -1.0, 0.0])
ctr.set_zoom(0.5)

# Run viewer
vis.run()
vis.destroy_window()