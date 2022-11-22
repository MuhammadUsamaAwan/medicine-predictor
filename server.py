from flask import Flask, request, render_template
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href="static/test.svg")
    else:
        medicine = request.form['medicine']
        steps = int(request.form['steps'])
        predict(medicine, steps)
        print('post')
        return render_template('index.html', href="static/test.svg")


def predict(medicine, step):
    data = pd.read_csv("./data.csv",parse_dates=['TRANSDATE'])
    value = data.loc[(data['PRIDESCR'] == medicine)]
    columns = ["NOMNCLTRE", "PRIDESCR", "STRENGTH", "TYPE", "DESCR", "AVGPRICE", "UNITPRICE", "UNIT"]
    value.drop(columns, inplace=True, axis=1)
    value = value.sort_values('TRANSDATE')
    value = value.groupby('TRANSDATE')['QUANTITY'].sum().reset_index()
    value = value.set_index(value['TRANSDATE'])
    y = value['QUANTITY'].resample('M').mean()
    mod = sm.tsa.statespace.SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_invertibility=False)
    results = mod.fit()
    pred_uc = results.get_forecast(steps=step)
    pred_ci = pred_uc.conf_int()
    ax = y.plot(label='observed', figsize=(18, 14))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('TRANSDATE')
    ax.set_ylabel('QUANTITY')
    plt.title("Predicted Graph for "+medicine)
    plt.legend()
    plt.savefig("./static/test.svg")
    plt.switch_backend('agg')
    return 'saved'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)