# importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from typing import List





def regressor_models_comparison(X_train, X_test, y_train, y_test, models_list: List[str], verbose: bool = True, visualize_fit: bool = False, visualize_resid: bool = False, round_prediction: int = None):
    """ Function creates comparison of metrics for the provided regression models"""

    if round_prediction is not None and not isinstance(round_prediction, int):
        raise TypeError("round_prediction must be an integer or None")

    if round_prediction is not None and round_prediction < 0:
        raise ValueError("round_prediction cannot be negative")


    # creating lists to store metrics
    models = models_list.copy()
    model_names = [m.__class__.__name__ for m in models]  # list with class names
    mse_list = []
    rmse_list = []
    mae_list = []
    r2_list = []
    model_list = []
    y_true = []
    y_pred_list = []

    # iterating through models
    for e, m in enumerate(models, 1):
        try:
            model = m
            # fitting model
            model.fit(X_train, y_train)

            # calculating predictions
            y_pred = model.predict(X_test)

            if round_prediction is not None:
                y_pred = np.round(y_pred, round_prediction)

   
            # evaluating
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            r2_list.append(r2)
            y_true.append(y_test)
            y_pred_list.append(y_pred)

            # saving metrics
            model_list.append(model_names[models_list.index(m)])

            if verbose: 
                max_len = len(max(model_names, key=len))
                models_num = len(models)
                print(f'Successfully evaluated ({e:2d} / {models_num}) : {model.__class__.__name__:{max_len}s} -> MAE: {mae:6.2f}, R2: {r2:.2f}')

        except Exception as e:
            print(f'Error evaluating {model.__class__.__name__}: {str(e)}')

    # creating df with results
    results = pd.DataFrame({"Model": model_list,
                            "MSE": mse_list,
                            "RMSE": rmse_list,
                            "MAE": mae_list,
                            "R2": r2_list})

    if visualize_fit:
        # plotting predicted vs true values for each model using plotly
        fig = go.Figure()
        
        min_y_true = min(y_true[0])
        max_y_true = max(y_true[0])

        for i, y_pred in enumerate(y_pred_list):
            fig.add_trace(go.Scatter(x=y_true[i], y=y_pred, mode='markers',
                                    name=model_names[i]))

        fig.add_trace(go.Scatter(x=[min_y_true, max_y_true], y= [min_y_true, max_y_true], mode='lines', 
                line=dict(color='black', width=2),name='Ideal prediction', showlegend=True) , )

        fig.update_layout(xaxis_title='True values', yaxis_title='Predicted values',
                        title={'text': 'Predicted vs True values (TEST DATA)', 'x': 0.5},
                        height=600)
        


        fig.show()
    

    if visualize_resid:
        # plotting residual vs true values for each model using plotly
        fig = go.Figure()

        for i, y_pred in enumerate(y_pred_list):
            resid = y_true[i] - y_pred
            fig.add_trace(go.Scatter(x=y_true[i], y=resid, mode='markers',
                                    name=model_names[i]),)

        fig.add_trace(go.Scatter(x=[min(y_true[i]), max(y_true[i])], y=[0, 0], mode='lines', 
                                    line=dict(color='gray', width=2), name='Zero line', showlegend=True, ))

    
        fig.update_layout(xaxis_title='True values', yaxis_title='Residuals',
                  title={'text': f'Residual plot ', 'x': 0.5},
                  height=600)

        
        fig.show()

    return results