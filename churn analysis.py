import sqlite3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snm
import warnings

warnings.filterwarnings('ignore')


conn = sqlite3.connect(r"C:\Users\pc\Downloads\customer_churn.db.db")


query = """
SELECT name
FROM sqlite_master
WHERE type='table';
"""

tables = pd.read_sql(query, conn)

print("Tables in the database:")
print(tables)


df_db_customer = pd.read_sql("SELECT * FROM db_customer", conn)
df_db_subscription = pd.read_sql("SELECT * FROM db_subscription", conn)
df_db_support = pd.read_sql("SELECT * FROM db_support", conn)


print("\nCustomer Table")
print(df_db_customer.head())

print("\nSubscription Table")
print(df_db_subscription.head())

print("\nSupport Table")
print(df_db_support.head())


print("\nCustomer Columns:")
print(df_db_customer.columns.tolist())

print("\nSubscription Columns:")
print(df_db_subscription.columns.tolist())

print("\nSupport Columns:")
print(df_db_support.columns.tolist())


df_db_customer.rename(columns={'name': 'customer_name'}, inplace=True)

print("\nColumns after renaming:")
print(df_db_customer.columns.tolist())


df_db_support.drop(df_db_support.columns[-2:], axis=1, inplace=True)

print("\nSupport columns after dropping:")
print(df_db_support.columns.tolist())


print("\nUnique Gender Values:")
print(df_db_customer['gender'].unique())

print("\nGender Counts:")
print(df_db_customer['gender'].value_counts())


state_country_mapping = (
    df_db_customer
    .dropna(subset=['country'])
    .set_index('state')['country']
    .to_dict()
)

df_db_customer['country'] = df_db_customer['country'].fillna(
    df_db_customer['state'].map(state_country_mapping)
)

print(df_db_customer[['state', 'country']])

print("\nMissing country values:")
print(df_db_customer['country'].isna().sum())


date_cols = ['subscription_start_date', 'renewal_date', 'cancellation_date']

for col in date_cols:
    df_db_subscription[col] = pd.to_datetime(df_db_subscription[col])

print(df_db_subscription[date_cols])

print(df_db_support.head())

df_db_support['complaint_date'] = pd.to_datetime(df_db_support['complaint_date'])
print(df_db_support.info())

#
df_db_subscription['churn_flag'] = np.where(df_db_subscription['cancellation_date'].notna(), 1, 0)
print(df_db_subscription.head())


df_db_support['escalations'] = df_db_support['escalations'].str.strip().str.upper()

df_db_support['complaint_count'] = df_db_support.groupby('customerid')['customerid'].transform('count')

df_support_agg = (
    df_db_support
    .sort_values('complaint_date')
    .drop_duplicates(subset=['customerid'], keep='last')  # keep each customer's most recent complaint
    [['customerid', 'complaint_date', 'escalations', 'complaint_count']]
    .rename(columns={'complaint_date': 'last_complaint_date'})
)

print(df_support_agg)


df = (
    df_db_subscription
    .merge(df_db_customer, on='customerid', how='left')
    .merge(df_support_agg, on='customerid', how='left')
)
print(df)

print(df_db_subscription.shape)


churn_rate = df['churn_flag'].mean() * 100
print(round(churn_rate, 2), "%")


retention_rate = 100 - churn_rate
print(round(retention_rate, 2), "%")


churn_by_plan = (
    df.groupby('plan_type')['churn_flag']
    .mean()
    .mul(100)
    .round(2)
    .reset_index(name='churn_rate_pct')
)
print(churn_by_plan)

# 5. ARPU - average revenue per
arpu = df['monthly_charges'].mean()
print("ARPU =", round(arpu, 2))

# 6. avg customer tenure
today = pd.Timestamp.today()
print(today)

df['tenure_days'] = np.where(
    df['cancellation_date'].notna(),
    (df['cancellation_date'] - df['subscription_start_date']).dt.days,
    (today - df['subscription_start_date']).dt.days
)
print(df.head())

avg_tenure = df['tenure_days'].mean()
print(avg_tenure)

# 7. revenue at risk - revenue lost from churned users
revenue_risk = df.loc[df['churn_flag'] == 1, 'monthly_charges'].sum()
print(revenue_risk)

# 8. escalation rate
escalation_rate = (df['escalations'] == 'Y').mean() * 100
print("escalation rate = ", round(escalation_rate, 2), "%")

# 9. avg complaints per user 0
avg_complaints = df['complaint_count'].fillna(0).mean()
print("avg complaints per user = ", round(avg_complaints, 2))

print(df.columns.tolist())

# 10. correlation: escalation vs chur
print(df[['escalations', 'churn_flag']].dropna())

df['escalations'] = np.where(df['escalations'] == 'Y', 1, 0)  # encode string to int

corr_df = df[['escalations', 'churn_flag']].dropna()
print(corr_df)

correlation = corr_df['escalations'].corr(df['churn_flag'])
print(correlation)

# 11. churn risk - bucket using churn_score
conditions = [
    df['churn_score'] >= 80,
    df['churn_score'] >= 50
]
choices = ['High', 'Medium']

df['churn_risk'] = np.select(conditions, choices, default='Low')

print(df[['churn_risk', 'churn_score']].head())


# Visualization
df_visual = df.copy()
print(df_visual.columns)

# 4.1 monthly churn trend (time-series KPI)
df_visual['cancellation_month'] = df_visual['cancellation_date'].dt.to_period('M')

churn_trend = (
    df_visual[df_visual['churn_flag'] == 1]
    .groupby('cancellation_month')
    .size()
)
print(churn_trend)

plt.figure(figsize=(8, 3))
plt.plot(
    churn_trend.index.astype(str),
    churn_trend.values,
    color='green',
    marker='o',
    linestyle='dashed',
    linewidth=2,
    markersize=8
)
plt.title('Churn Trend')
plt.xlabel('Months')
plt.ylabel('Number of Churns')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4.2 churn by plan type
churn_plan = df_visual.groupby('plan_type')['churn_flag'].mean()

colors = plt.cm.Set2(np.linspace(0, 1, len(churn_plan)))

plt.figure(figsize=(8, 6))
plt.bar(churn_plan.index, churn_plan.values, color=colors)
plt.title('Churn Rate by Plan Type')
plt.xlabel('Plan Type')
plt.ylabel('Churn Rate')
plt.show()


# Visualization s

import seaborn as sns

df_encoded = df_visual[['plan_type', 'contract_type', 'churn_score', 'churn_flag', 'churn_risk', 'escalations']].copy()

categorical_cols = ['plan_type', 'contract_type', 'churn_risk']

for col in categorical_cols:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

plt.figure(figsize=(10, 6))
snm.heatmap(df_encoded.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Ordinal encoding example
df_encoded_ord = df_visual[['plan_type', 'contract_type', 'churn_score', 'churn_flag', 'churn_risk', 'escalations']].head().copy()

order_mappings = {
    'plan_type': ['Basic', 'Standard', 'Premium'],
    'contract_type': ['Monthly', 'Annual'],
    'churn_risk': ['Low', 'Medium', 'High'],
}

for col, order in order_mappings.items():
    df_encoded_ord[col] = pd.Categorical(df_encoded_ord[col], categories=order, ordered=True)

print(df_encoded_ord.head())


print(sns.pairplot(df_encoded))




#catplt/Facegrid plot

sns.catplot(data=df_visual,
            x='plan_type',
            y='monthly_charges',
            hue='gender',
            col='churn_risk'
            )
print(sns.catplot(data=df_visual,))



#create db in sql
import sqlite3

# connect
conn = sqlite3.connect('tes-database.sqlite')

# create table
conn.execute("DROP TABLE IF EXISTS users")
conn.execute("CREATE TABLE users (first_name TEXT, country TEXT, budget INTEGER)")


# insert data
cursor = conn.cursor()
cursor.execute("""
INSERT INTO users VALUES
    ('madhav','india',5000),
    ('rishabh','germany',2500),
    ('riya','india',25000)
""")

# commit changes
conn.commit()

# check data
for row in cursor.execute("SELECT * FROM users"):
    print(row)


# check inserted data in table

conn = sqlite3.connect('tes-database.sqlite')

query = "SELECT * FROM users"

df_results = pd.read_sql_query(query, conn)
print(df_results.head())

#aggageration

query = """
   SELECT country,sum(budget) as total_budget
   FROM users
   GROUP BY country
"""
df.agg = pd.read_sql_query(query, conn)
df_results = pd.read_sql_query(query, conn)







conn.close()