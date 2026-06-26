import pandas as pd

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset bruto de mensagens SMS.
    
    :param filepath: Caminho do arquivo de texto bruto.
    :return: DataFrame contendo as colunas 'label' e 'mensagem'.
    """
    df = pd.read_csv(filepath, sep='\t', names=["label", "mensagem"])
    return df