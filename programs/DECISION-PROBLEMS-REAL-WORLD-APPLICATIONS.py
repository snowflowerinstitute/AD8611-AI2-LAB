import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

customer_dataset = pd.read_csv('../datasets/3-mall-customers.csv')
customer_dataset.head()
print(customer_dataset.shape)
customer_dataset.describe()
print(customer_dataset.dtypes)
customer_dataset.info()

customer_dataset.isnull().sum()

customer_dataset.drop(['CustomerID'], axis=1, inplace=True)
customer_dataset.head()

plt.figure(1, figsize=(16, 5))
n = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(customer_dataset[x], bins=20, kde=True)
    plt.axvline(customer_dataset[x].mean(), color='red', linestyle='--')
    plt.title('Dist plot of {}'.format(x))
plt.show()

plt.figure(figsize=(16, 5))
sns.countplot(y='Genre', data=customer_dataset)
plt.title('Bar Graph')
plt.show()

plt.figure(1, figsize=(15, 7))
n = 0
for cols in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    sns.set(style='whitegrid')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.violinplot(x=cols, y='Genre', data=customer_dataset)
    plt.ylabel('Genre' if n == 1 else '')
    plt.title('Violin Plot')
plt.show()

age_18_25 = customer_dataset.Age[(customer_dataset.Age >= 18) & (customer_dataset.Age <= 25)]
age_26_35 = customer_dataset.Age[(customer_dataset.Age >= 26) & (customer_dataset.Age <= 35)]
age_36_45 = customer_dataset.Age[(customer_dataset.Age >= 36) & (customer_dataset.Age <= 45)]
age_46_55 = customer_dataset.Age[(customer_dataset.Age >= 46) & (customer_dataset.Age <= 55)]
age_above_55 = customer_dataset.Age[(customer_dataset.Age >= 56)]

agex = ['18-25', '26-35', '36-45', '46-55', '55+']
agey = [len(age_18_25.values), len(age_26_35.values), len(age_36_45.values), len(age_46_55.values),
        len(age_above_55.values)]

plt.figure(figsize=(15, 6))
sns.barplot(x=agex, y=agey, palette='mako')
plt.title('Bar Plot of Age')
plt.xlabel('Age')
plt.ylabel('Number of Customer')
plt.show()

sns.relplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=customer_dataset)
plt.title('Scatter Plot of Annual Income vs Spending Score')
plt.show()

ss_1_20 = customer_dataset['Spending Score (1-100)'][
    (customer_dataset['Spending Score (1-100)'] >= 1) & (customer_dataset['Spending Score (1-100)'] <= 20)]
ss_21_40 = customer_dataset['Spending Score (1-100)'][
    (customer_dataset['Spending Score (1-100)'] >= 21) & (customer_dataset['Spending Score (1-100)'] <= 40)]
ss_41_60 = customer_dataset['Spending Score (1-100)'][
    (customer_dataset['Spending Score (1-100)'] >= 41) & (customer_dataset['Spending Score (1-100)'] <= 60)]
ss_61_80 = customer_dataset['Spending Score (1-100)'][
    (customer_dataset['Spending Score (1-100)'] >= 61) & (customer_dataset['Spending Score (1-100)'] <= 80)]
ss_81_100 = customer_dataset['Spending Score (1-100)'][
    (customer_dataset['Spending Score (1-100)'] >= 81) & (customer_dataset['Spending Score (1-100)'] <= 100)]
ssx = ['1-20', '21-40', '41-60', '61-80', '81-100']
ssy = [len(ss_1_20.values), len(ss_21_40.values), len(ss_41_60.values), len(ss_61_80.values), len(ss_81_100.values)]

plt.figure(figsize=(15, 6))
sns.barplot(x=ssx, y=ssy, palette='rocket')
plt.title('Bar Plot of Spending Score')
plt.xlabel('Score')
plt.ylabel('Number of Customer having the Score')
plt.show()

ann_0_30 = customer_dataset['Annual Income (k$)'][
    (customer_dataset['Annual Income (k$)'] >= 0) & (customer_dataset['Annual Income (k$)'] <= 30)]
ann_31_60 = customer_dataset['Annual Income (k$)'][
    (customer_dataset['Annual Income (k$)'] >= 31) & (customer_dataset['Annual Income (k$)'] <= 60)]
ann_61_90 = customer_dataset['Annual Income (k$)'][
    (customer_dataset['Annual Income (k$)'] >= 61) & (customer_dataset['Annual Income (k$)'] <= 90)]
ann_91_120 = customer_dataset['Annual Income (k$)'][
    (customer_dataset['Annual Income (k$)'] >= 91) & (customer_dataset['Annual Income (k$)'] <= 120)]
ann_121_150 = customer_dataset['Annual Income (k$)'][
    (customer_dataset['Annual Income (k$)'] >= 121) & (customer_dataset['Annual Income (k$)'] <= 150)]

annx = ['$ 0-30,000', '$ 31,000-60,000', '$ 61,000-90,000', '$ 91,000-1,20,000', '$ 1,21,000-1,50,000']
anny = [len(ann_0_30.values), len(ann_31_60.values), len(ann_61_90.values), len(ann_91_120.values),
        len(ann_121_150.values)]

plt.figure(figsize=(15, 6))
sns.barplot(x=annx, y=anny, palette='Spectral')
plt.title('Bar Plot of Annual Income')
plt.xlabel('Income')
plt.ylabel('Number of Customer')
plt.show()

X1 = customer_dataset.loc[:, ['Age', 'Spending Score (1-100)']].values

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color='red', marker='8')
plt.title('Graph of K Value vs WCSS')
plt.xlabel('K Value')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, n_init=10)
label = kmeans.fit_predict(X1)
print(label)
print(kmeans.cluster_centers_)
plt.scatter(X1[:, 0], X1[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
plt.title('Graph of Age vs Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()

X2 = customer_dataset.loc[:, ['Annual Income (k$)', 'Spending Score (1-100)']].values

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color='red', marker='8')
plt.title('Graph of K Value vs WCSS')
plt.xlabel('K Value')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, n_init=10)
label = kmeans.fit_predict(X2)
print(label)
print(kmeans.cluster_centers_)
plt.scatter(X2[:, 0], X2[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
plt.title('Graph of Annual Income vs Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')
plt.show()

X3 = customer_dataset.iloc[:, 1:]
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss, linewidth=2, color='red', marker='8')
plt.title('Graph of WCSS vs K Value')
plt.xlabel('K Value')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, n_init=10)
label = kmeans.fit_predict(X3)
print(label)
print(kmeans.cluster_centers_)

clusters = kmeans.fit_predict(X3)
customer_dataset['label'] = clusters

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(customer_dataset.Age[customer_dataset.label == 0],
           customer_dataset['Annual Income (k$)'][customer_dataset.label == 0],
           customer_dataset['Spending Score (1-100)'][customer_dataset.label == 0], c='blue', s=60)
ax.scatter(customer_dataset.Age[customer_dataset.label == 1],
           customer_dataset['Annual Income (k$)'][customer_dataset.label == 1],
           customer_dataset['Spending Score (1-100)'][customer_dataset.label == 1], c='red', s=60)
ax.scatter(customer_dataset.Age[customer_dataset.label == 2],
           customer_dataset['Annual Income (k$)'][customer_dataset.label == 2],
           customer_dataset['Spending Score (1-100)'][customer_dataset.label == 2], c='green', s=60)
ax.scatter(customer_dataset.Age[customer_dataset.label == 3],
           customer_dataset['Annual Income (k$)'][customer_dataset.label == 3],
           customer_dataset['Spending Score (1-100)'][customer_dataset.label == 3], c='orange', s=60)
ax.scatter(customer_dataset.Age[customer_dataset.label == 4],
           customer_dataset['Annual Income (k$)'][customer_dataset.label == 4],
           customer_dataset['Spending Score (1-100)'][customer_dataset.label == 4], c='purple', s=60)
ax.view_init(30, 185)

plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.title('Customer Clusters')
ax.set_zlabel('Spending Score (1-100)')
plt.show()
