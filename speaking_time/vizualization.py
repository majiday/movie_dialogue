import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar_chart(contingency_table):
    plt.figure(figsize=(12, 8))
    contingency_table.plot(kind='bar', stacked=True)
    plt.title('Number of Lines Spoken by Each Character in Each Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Lines')
    plt.legend(title='Character')
    plt.show()

def plot_heatmap(contingency_table):
    plt.figure(figsize=(12, 8))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Heatmap of Lines Spoken by Each Character in Each Season')
    plt.xlabel('Character')
    plt.ylabel('Season')
    plt.show()
