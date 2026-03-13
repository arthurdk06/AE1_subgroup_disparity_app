import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# Regions to exclude from London borough-level analysis (aggregates, not boroughs)
_NON_BOROUGH_REGIONS = [
    "South East", "South West", "England",
    "London", "West Midlands", "East Midlands",
    "Yorkshire and the Humber", "North West",
    "North East", "Inner London", "Outer London",
    "East of England",
]


class PreprocessingStep(ABC):

    @abstractmethod
    def process(self) -> pd.DataFrame:
        pass


class TotalProcessing(PreprocessingStep):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def process(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_name, header=1)
        
        new_column_names = [
        'region_id', 'region_name', 'all_total_pupils', 'all_attainment_8_score',
        'all_progress_8_score', 'all_entered_english_maths', 'all_achieving_grade_5_or_above_eng_math',
        'all_achieving_grade_4_or_above_eng_math', 'boys_total_pupils', 'boys_attainment_8_score',
        'boys_progress_8_score', 'boys_entered_components', 'boys_achieving_grade_5_or_above_overall',
        'boys_achieving_grade_4_or_above_overall', 'girls_total_pupils', 'girls_attainment_8_score',
        'girls_progress_8_score', 'girls_entered_components', 'girls_achieving_grade_5_or_above_overall',
        'girls_achieving_grade_4_or_above_overall'
        ]

        df.columns = new_column_names
        df_cleaned = df.dropna(how='all')
        df_cleaned = df_cleaned.dropna(subset=['region_name'])
        df_cleaned = df_cleaned.dropna(subset=['all_total_pupils'])

        for col in ['all_total_pupils', 'boys_total_pupils', 'girls_total_pupils']:
            df_cleaned[col] = df_cleaned[col].str.replace(',', '', regex=False).astype(int)

        for col in ['all_progress_8_score', 'boys_progress_8_score', 'girls_progress_8_score']:
            df_cleaned[col] = df_cleaned[col].astype(float)

        proportional_cols = [
            'boys_entered_components',
            'boys_achieving_grade_5_or_above_overall',
            'boys_achieving_grade_4_or_above_overall',
            'girls_entered_components',
            'girls_achieving_grade_5_or_above_overall',
            'girls_achieving_grade_4_or_above_overall'
        ]
        for col in proportional_cols:
            df_cleaned[col] = df_cleaned[col].astype(float)

        df_cleaned = df_cleaned[~df_cleaned["region_name"].isin(_NON_BOROUGH_REGIONS)]

        return df_cleaned



class SENProcessing(PreprocessingStep):

    def __init__(self, file_name: str):
        self.file_name = file_name

    def process(self) -> pd.DataFrame:
        df_sen_raw = pd.read_csv(self.file_name, header=1)
        
        new_column_names = []

        group_definitions = [
            {'name': 'total', 'start_index': 2}, 
            {'name': 'boys', 'start_index': 14},
            {'name': 'girls', 'start_index': 26} 
        ]

        metric_short_names = [
            'num_pupils_ks4',
            'attainment_8',
            'progress_8'
        ]

        sen_categories = [
            'total',       
            'no_sen',
            'sen_state_ehc',
            'sen_supp'
        ]

        for i in range(len(df_sen_raw.columns)):
            if i == 0:
                new_column_names.append('region_id')
                continue
            if i == 1:
                new_column_names.append('region_name')
                continue

            column_named = False
            for group_def in group_definitions:
                group_name = group_def['name']
                group_start = group_def['start_index']

                if i >= group_start and i < group_start + (len(metric_short_names) * len(sen_categories)):
                    offset_in_group = i - group_start

                    metric_idx = offset_in_group // len(sen_categories)
                    metric_name = metric_short_names[metric_idx]

                    sen_cat_idx = offset_in_group % len(sen_categories)
                    sen_cat_name = sen_categories[sen_cat_idx]

                    new_column_names.append(f"{group_name}_{metric_name}_{sen_cat_name}")
                    column_named = True
                    break

            if not column_named:
                new_column_names.append(f"unhandled_col_{i}")

        df_sen = df_sen_raw.copy()
        df_sen.columns = new_column_names
        df_sen = df_sen.drop([0, 1, 2, 3]).reset_index(drop=True)
        return df_sen
            


class THProcessing(PreprocessingStep):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def process(self) -> pd.DataFrame:
        df_raw = pd.read_csv(self.file_name)

        df_cleaned = df_raw[
            [
                'SCHNAME',
                'TPUP',
                'BPUP',
                'GPUP',
                'ATT8SCR',
                'SENE4',
                'SENK4',
                'SEN_ALL4',
                'ATT8SCR_GIRLS',
                'ATT8SCR_BOYS',
                'P8MEA',
            ]
        ].copy()
        df_cleaned = df_cleaned[df_cleaned['TPUP'] > 10]
        df_cleaned.replace(['NE', 'SUPP', 'NP'], np.nan, inplace=True)
        df_cleaned.dropna(subset=['SCHNAME'], inplace=True)

        new_column_names = {
            'SCHNAME': 'school_name',
            'TPUP': 'total_pupils',
            'BPUP': 'boys_pupils',
            'GPUP': 'girls_pupils',
            'ATT8SCR': 'attainment_8_score_all',
            'SENE4': 'sen_ehc_plan_pupils',
            'SENK4': 'sen_support_pupils',
            'SEN_ALL4': 'sen_total_pupils',
            'ATT8SCR_GIRLS': 'attainment_8_score_girls',
            'ATT8SCR_BOYS': 'attainment_8_score_boys',
            'P8MEA': 'progress_8_score'
        }

        df_cleaned.rename(columns=new_column_names, inplace=True)

        pupil_cols = ['total_pupils', 'boys_pupils', 'girls_pupils']
        for col in pupil_cols:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0).astype(int)

        attainment_sen_cols = [
            'attainment_8_score_all', 'sen_ehc_plan_pupils', 'sen_support_pupils', 'sen_total_pupils',
            'attainment_8_score_girls', 'attainment_8_score_boys'
        ]
        for col in attainment_sen_cols:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        df_cleaned.dropna(subset=['attainment_8_score_all'], inplace=True)

        df_cleaned.sort_values(by='attainment_8_score_all', inplace=True)

        return df_cleaned


def _zscore(series: pd.Series, value: float) -> Optional[float]:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return None
    return (value - mean) / std


def load_gender_borough(file_path: str) -> pd.DataFrame:
    return TotalProcessing(file_path).process()


def load_sen_borough(file_path: str) -> pd.DataFrame:
    df = SENProcessing(file_path).process()
    return df


def load_th_schools(file_path: str) -> pd.DataFrame:
    return THProcessing(file_path).process()


def compute_school_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['gender_gap'] = df['attainment_8_score_girls'] - df['attainment_8_score_boys']
    df['sen_proportion'] = df['sen_total_pupils'] / df['total_pupils']
    return df


def find_data_files(data_dir: str) -> dict[str, list[Path]]:
    data_path = Path(data_dir)
    return {
        'gender': sorted(data_path.glob('GCSE results by sex - *.csv')),
        'sen': sorted(data_path.glob('GCSE results by SEN - *.csv')),
        'th': sorted(data_path.glob('TH results *.csv')),
    }


def compute_gender_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['gender_gap'] = df['girls_attainment_8_score'] - df['boys_attainment_8_score']
    gap_mean = df['gender_gap'].mean()
    gap_std = df['gender_gap'].std(ddof=0)
    df['gender_gap_zscore_london'] = (df['gender_gap'] - gap_mean) / gap_std if gap_std else np.nan
    return df


def compute_sen_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Filter to London boroughs only (consistent with gender analysis)
    if "region_name" in df.columns:
        df = df[~df["region_name"].isin(_NON_BOROUGH_REGIONS)].copy()

    no_sen_count = pd.to_numeric(df["total_num_pupils_ks4_no_sen"], errors="coerce")
    ehc_count = pd.to_numeric(df['total_num_pupils_ks4_sen_state_ehc'], errors='coerce')
    supp_count = pd.to_numeric(df['total_num_pupils_ks4_sen_supp'], errors='coerce')
    total_count = pd.to_numeric(df['total_num_pupils_ks4_total'], errors='coerce')

    no_sen_att8 = pd.to_numeric(df['total_attainment_8_no_sen'], errors='coerce')
    ehc_att8 = pd.to_numeric(df['total_attainment_8_sen_state_ehc'], errors='coerce')
    supp_att8 = pd.to_numeric(df['total_attainment_8_sen_supp'], errors='coerce')

    sen_total_count = ehc_count + supp_count
    sen_att8 = (ehc_att8 * ehc_count + supp_att8 * supp_count) / sen_total_count

    df['sen_total_pupils'] = sen_total_count
    df['sen_proportion'] = sen_total_count / total_count
    df['sen_gap'] = no_sen_att8 - sen_att8

    gap_mean = df['sen_gap'].mean()
    gap_std = df['sen_gap'].std(ddof=0)
    df['sen_gap_zscore_london'] = (df['sen_gap'] - gap_mean) / gap_std if gap_std else np.nan

    return df