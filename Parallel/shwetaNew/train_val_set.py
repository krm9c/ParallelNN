from pandas import DataFrame


class TrainValSet:
    def __init__(self, df_train= DataFrame, df_val= DataFrame):
        self.x_train, self.y_train = self.get_x_y(df_train)
        self.x_val, self.y_val = self.get_x_y(df_val)

    def get_x_y(self, df= DataFrame):
        X = df.iloc[:, 0:-1].values
        y = df.iloc[:, -1].values

        return X, y