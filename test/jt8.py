import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def fun1():
    plt.plot([1, 2, 3], label=['Inline label', 'test'])
    plt.legend()
    plt.show()

def fun2():
    df = pd.DataFrame()
    df['Classes'] = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df['Accuracy'] = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sns.barplot(data=df, x='Classes', y='Accuracy')
    plt.show()

if __name__ == '__main__':
    fun2()