import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def identify_columns(df):
    num_cols = []
    cat_cols = []
    for col in df.columns.to_list():
        if df[col].dtype == "object":
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols

def Handling_NULL_Values(df):
    
    
    nulls_df = df.isna().sum().reset_index().rename(columns={0: "Nulls_Count"})
    nulls_df = nulls_df[nulls_df["Nulls_Count"] > 0].sort_values(
        by="Nulls_Count", ascending=False
    )
    
    nulls_df = nulls_df.set_index("index", drop=True)

    feature_to_be_dropped = []
    feature_to_be_filled = []
    rows_to_be_dropped = []   
    for col in nulls_df.index:
            if nulls_df["Nulls_Count"][col] >= df.shape[0] * (60 / 100) - 10:
                feature_to_be_dropped.append(col)
            elif nulls_df["Nulls_Count"][col] < df.shape[0] * (3 / 100) - 10:
                rows_to_be_dropped.append(col)
            else:
                feature_to_be_filled.append(col)   
                

    df.drop(axis=1, columns=feature_to_be_dropped, inplace=True)

    df.dropna(axis=0, subset=rows_to_be_dropped, inplace=True)

    for col in feature_to_be_filled:
        if df[col].dtype == "object":  # Check if column is categorical
            df[col] = df[col].fillna(df[col].mode()[0])  # Fill missing values with mode
        else:
            df[col] = df[col].fillna(df[col].mean())
            
    return df


# case_1

def drop_highly_uniform_columns(df,cat_col):
    for col in cat_col: 
        value_counts = df[col].value_counts()
        max_count = value_counts.max()
        total_count = df.shape[0]  # Total count of all values in the DataFrame
        if max_count >= total_count * 0.8:
            df.drop(columns=col, inplace=True)  # Drop the column

    return df

# case_2

# def drop_highly_uniform_columns(df, cat_col):
#     for col in cat_col:
#         value_counts = df[col].value_counts()
#         max_count = value_counts.max()
#         total_count = df.shape[0]  # Total count of all values in the DataFrame
#         if max_count >= total_count * 0.8:
#             if len(df[col].unique()) > 1:
#                 dummies = pd.get_dummies(df[col], prefix=col)
#                 df = pd.concat([df, dummies], axis=1)
#                 df.drop(columns=[col], inplace=True)

#     return df

# case_3

# def oversample_class(df, cat_col):
    
#     for col in cat_col: 
#         value_counts = df[col].value_counts()
#         max_count = value_counts.max()
#         total_count = df.shape[0]  # Total count of all values in the DataFrame
#         if max_count >= total_count * 0.8:
#             oversample = RandomOverSampler()
#             X_resampled, y_resampled = oversample.fit_resample(df.drop(columns=[col]), df[col])
#             df = pd.concat([X_resampled, y_resampled], axis=1)
#             break
#     return df


def drop_highly_correlated_features(df, num_cols):
    
    
    corr_matrix = df[num_cols].corr()

    columns_to_drop = []

    for row_idx in range(corr_matrix.shape[0]):

        for col_idx in range(row_idx + 1, corr_matrix.shape[0]):

            if np.abs(corr_matrix.iloc[row_idx, col_idx]) > 0.7:

                var_row_corr_with_other = (
                    np.abs(df[num_cols].corr()[num_cols[row_idx]])
                    .drop([num_cols[row_idx], num_cols[col_idx]])
                    .max()
                )

                var_col_corr_with_other = (
                    np.abs(df[num_cols].corr()[num_cols[col_idx]])
                    .drop([num_cols[col_idx], num_cols[col_idx]])
                    .max()
                )

                if var_row_corr_with_other < var_col_corr_with_other:
                    columns_to_drop.append(num_cols[col_idx])
                else:
                    columns_to_drop.append(num_cols[row_idx])

    df.drop(columns=columns_to_drop,inplace=True)
    
    return df

def Outliers_Handling(data, num_cols):

    for col in num_cols:
        Q1 = np.quantile(data[col], 0.25)
        Q3 = np.quantile(data[col], 0.75)
        IQR = Q3 - Q1

        Upper_Bound = Q3 + 1.5 * IQR
        Lower_Bound = Q1 - 1.5 * IQR
        
        data[col] = np.clip(data[col], Lower_Bound, Upper_Bound)

    return data

def skewness_calc(col, df):
    std_col = np.std(df[col])
    mean_col = np.mean(df[col])
    size_rows = df.shape[0]
    diff_col = df[col] - mean_col
    diff_col_powered = np.power(diff_col, 3)
    skewness = np.sum(diff_col_powered) / ((size_rows - 1) * np.power(std_col, 3))
    return skewness

def skewed_data_transformation(value, skewness):
    if abs(skewness) >= 1 and value >= 0:
        return np.log1p(value)
    else:
        return value
    

def min_max_scale(df, num_cols):
        
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    return df_scaled

def determine_categorical_type(column):
    unique_values = column.unique()
    num_unique_values = len(unique_values)

    if num_unique_values <= 2:
        return "nominal"
    elif num_unique_values <= 5:
        return "ordinal"
    elif num_unique_values > 5 and num_unique_values <= 20:
        if all(value.isdigit() for value in unique_values):
            return "ordinal"
        else:
            return "nominal"
    else:
        return "col_to_be_drop"
     

def encode_ordinal_columns(df, ordinal_cols):
    encoded_df = df.copy()
    label_encoder = LabelEncoder()
    for col in ordinal_cols:
        encoded_df[col] = label_encoder.fit_transform(df[col])
    return encoded_df


def encode_nominal_columns(df, nominal_cols):
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[nominal_cols])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(nominal_cols))
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    # Drop the original nominal columns
    df_encoded = df_encoded.drop(nominal_cols, axis=1)

    return df_encoded


def encode_categorical_columns(df, cat_cols):
    ordinal_cols = []
    nominal_cols = []
    col_to_be_drop = []
    for col in cat_cols:
        col_type = determine_categorical_type(df[col])
        if col_type == "ordinal":
            ordinal_cols.append(col)
        elif col_type == "nominal":
            nominal_cols.append(col)
        else :
            col_to_be_drop.append(col)
            
    encoded_df = df.copy()
    if ordinal_cols:
        encoded_df = encode_ordinal_columns(encoded_df, ordinal_cols)
    if nominal_cols:
        encoded_df = encode_nominal_columns(encoded_df, nominal_cols)
    if  col_to_be_drop:
        encoded_df.drop(columns=col_to_be_drop,inplace=True)
    return encoded_df



def unsupervised_preprocessing(df):

    df = Handling_NULL_Values(df)

    num_cols, cat_cols = identify_columns(df)
    df = drop_highly_uniform_columns(df, cat_cols)

    # num_cols, cat_cols = identify_columns(df)
    # df = Handling_highly_uniform_columns(df, cat_cols)

    # num_cols, cat_cols = identify_columns(df)
    # df = oversample_Handling_highly_uniform_columns(df, cat_cols)

    num_cols, cat_cols = identify_columns(df)
    df = drop_highly_correlated_features(df, num_cols)

    num_cols, cat_cols = identify_columns(df)
    df = Outliers_Handling(df, num_cols)

    num_cols, cat_cols = identify_columns(df)
    for col in num_cols:
        sk = skewness_calc(col, df)
        df[col] = df[col].apply(lambda x: skewed_data_transformation(x, sk))

    num_cols, cat_cols = identify_columns(df)
    df = min_max_scale(df, num_cols)


    num_cols, cat_cols = identify_columns(df)
    df = encode_categorical_columns(df, cat_cols)

    return df


