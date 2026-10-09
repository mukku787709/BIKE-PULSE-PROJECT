"""Chronological model comparison with separate selection and calibration sets."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bikepulse.data import FEATURES, download, wrangle

def metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y,pred)), "RMSE": float(np.sqrt(mean_squared_error(y,pred))), "R2": float(r2_score(y,pred))}

def run(epochs=80, output="artifacts"):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    df = wrangle(download())
    n = len(df)
    a,b,c = int(n*.6),int(n*.75),int(n*.85)
    train,val,cal,test = df.iloc[:a],df.iloc[a:b],df.iloc[b:c],df.iloc[c:]
    x,y = train[FEATURES], train.cnt
    models = {"Mean baseline": DummyRegressor(), "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)), "Random forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42,n_jobs=2))}
    validation = {}
    predict = {}
    for name,model in models.items():
        model.fit(x,y)
        predict[name] = lambda frame, model=model: np.maximum(0, model.predict(frame[FEATURES]))
        validation[name] = metrics(val.cnt, predict[name](val))
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    tx = torch.tensor(prep.fit_transform(x),dtype=torch.float32)
    vx = torch.tensor(prep.transform(val[FEATURES]),dtype=torch.float32)
    target_mean,target_std = float(y.mean()),float(y.std())
    ty = torch.tensor(((y.to_numpy()-target_mean)/target_std)[:,None],dtype=torch.float32)
    net = nn.Sequential(nn.Linear(len(FEATURES),64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    optimizer = torch.optim.Adam(net.parameters(),lr=.002)
    best_loss,best_state = float("inf"),None
    history=[]
    for epoch in range(epochs):
        net.train()
        for batch in torch.randperm(len(tx)).split(256):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(net(tx[batch]),ty[batch])
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            vp = np.maximum(0,net(vx).numpy().ravel()*target_std+target_mean)
        score = mean_absolute_error(val.cnt,vp)
        history.append({"epoch":epoch+1,"validation_mae":float(score)})
        if score < best_loss:
            best_loss,best_state = score,copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    def neural_predict(frame):
        net.eval()
        with torch.no_grad():
            return np.maximum(0,net(torch.tensor(prep.transform(frame[FEATURES]),dtype=torch.float32)).numpy().ravel()*target_std+target_mean)
    predict["Neural network"] = neural_predict
    validation["Neural network"] = metrics(val.cnt,neural_predict(val))
    winner = min(validation,key=lambda k:validation[k]["MAE"])
    # Calibration is disjoint from selection and test; temporal drift can affect coverage.
    residual = np.abs(cal.cnt.to_numpy()-predict[winner](cal))
    q = float(np.quantile(residual, min(1,np.ceil((len(residual)+1)*.9)/len(residual)),method="higher"))
    test_pred = predict[winner](test)
    lower,upper = np.maximum(0,test_pred-q),test_pred+q
    anomaly_prep = make_pipeline(SimpleImputer(strategy="median"),StandardScaler())
    detector = IsolationForest(contamination=.03,random_state=42).fit(anomaly_prep.fit_transform(x))
    anomalies = detector.predict(anomaly_prep.transform(test[FEATURES])) == -1
    report = {"rows":n,"features":FEATURES,"winner":winner,"validation":validation,"test":{k:metrics(test.cnt,p(test)) for k,p in predict.items()},"interval":{"nominal_coverage":.9,"observed_test_coverage":float(np.mean((test.cnt>=lower)&(test.cnt<=upper))),"radius":q},"splits":{k:{"rows":len(v),"start":str(v.timestamp.min()),"end":str(v.timestamp.max())} for k,v in {"train":train,"validation":val,"calibration":cal,"test":test}.items()}}
    path=Path(output)
    path.mkdir(parents=True,exist_ok=True)
    (path/"metrics.json").write_text(json.dumps(report,indent=2))
    pd.DataFrame({"timestamp":test.timestamp,"actual":test.cnt,"prediction":test_pred,"lower":lower,"upper":upper,"anomaly":anomalies}).to_csv(path/"predictions.csv",index=False)
    forest=models["Random forest"][-1]
    pd.DataFrame({"feature":FEATURES,"importance":forest.feature_importances_}).sort_values("importance",ascending=False).to_csv(path/"importance.csv",index=False)
    pd.DataFrame(history).to_csv(path/"training_history.csv",index=False)
    print(json.dumps(report,indent=2))
    return report

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=80)
    args=parser.parse_args()
    run(epochs=args.epochs)
