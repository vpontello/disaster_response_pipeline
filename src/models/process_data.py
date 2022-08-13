import sys
import pandas as pd
import sqlite3


def load_data(messages_filepath, categories_filepath):
    '''
    this function loads the data from the csv files
    '''
    df_messages = pd.read_csv('../../data/01_raw/'+messages_filepath)
    df_categories = pd.read_csv('../../data/01_raw/'+categories_filepath)
    df = df_messages.merge(df_categories, on='id')
    return df


def clean_data(df):
    '''
    This function receives the df and transforms the data, in order to clean and format the columns.
    the output is the base dataset for training the models.
    '''
    # extract a list of new column names for categories from the first row.
    categories_cols = [col.split('-')[0] for col in df['categories'].iloc[0].split(';')]
    # create a dataframe of the 36 individual category columns
    df_expanded_cetegories = df['categories'].str.split(';',expand=True)
    # rename the columns of `categories`
    df_expanded_cetegories.columns = categories_cols
    # set each value to be the last character of the string and convert column from string to numeric
    for col in categories_cols:
        df_expanded_cetegories[col] = df_expanded_cetegories[col].apply(lambda s: int(s.split('-')[-1]))

    # concatenate the extracted and cleaned categories with the former dataset
    df = pd.concat([df, df_expanded_cetegories], axis=1)
    # drop categories columns (since it`s content is alredy on the concatenated columns)
    df.drop('categories', axis=1, inplace=True)

    # checking duplicates
    print(f'There are {sum(df.duplicated())} duplicates on {df.shape[0]} rows')
    # drop duplicates
    print('Drop duplicates')
    df.drop_duplicates(inplace=True)
    print(f'Now, there are {sum(df.duplicated())} duplicates on {df.shape[0]} rows')

    return df


def save_data(df, database_filename='disaster.db'):
    'saving the data in a SQLite DB'
    # create SQLite DB
    conn = sqlite3.connect('../../data/02_trusted/'+database_filename)
    # export clean dataset to SQLite
    df.to_sql('messages_dataset',con=conn,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'messages.csv categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()