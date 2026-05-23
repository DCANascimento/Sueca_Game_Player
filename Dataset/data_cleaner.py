import ast
import csv
import json
import re
import sys
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

trump_map = {'♣': 0, '♦': 1, '♥': 2, '♠': 3}
position_map = {'NORTH': 0, 'WEST': 1, 'SOUTH': 2, 'EAST': 3}
output_base_columns = [
    'game_number',
    'round_number',
    'round_points',
    'round_winner_team',
    'team1_before',
    'team2_before',
    'team1_after',
    'team2_after',
]

def parse_cards_played(text):
    if not text:
        return []
    plays = [p.strip() for p in str(text).split('|')]
    result = []
    for play in plays:
        m = re.search(r'Positions\.([A-Z]+):(\d+)', play)
        if m:
            position = position_map.get(m.group(1))
            card = int(m.group(2))
            result.append((position, card))
    return result

def parse_round_actions(text):
    if not text:
        return []
    normalized = str(text).strip()
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(normalized)
        except (ValueError, SyntaxError, TypeError):
            return []

def to_bits(cards):
    bits = [0] * 40
    if isinstance(cards, str):
        try:
            cards = ast.literal_eval(cards)
        except (ValueError, SyntaxError):
            return bits
    if isinstance(cards, list):
        for card in cards:
            try:
                card = int(card)
            except (TypeError, ValueError):
                continue
            if 0 <= card < 40:
                bits[card] = 1
    return bits

def encode_suit(suit):
    if not suit:
        return None
    return trump_map.get(suit)

def first_non_null_suit(round_actions):
    for action in round_actions:
        suit = action.get('lead_suit')
        if suit is not None:
            return encode_suit(suit)
    return None

def get_round_action(actions, index):
    if len(actions) > index:
        return actions[index]
    return {}

def action_to_bits(action, field_name):
    if not action:
        return [0] * 40
    return to_bits(action.get(field_name, []))

def parse_int(value):
    if value in (None, ''):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value

def build_clean_row(row):
    cards_played_parsed = parse_cards_played(row.get('cards_played'))
    round_actions_parsed = parse_round_actions(row.get('round_actions'))

    cleaned_row = {
        'game_number': parse_int(row.get('game_number')),
        'round_number': parse_int(row.get('round_number')),
        'round_points': parse_int(row.get('round_points')),
        'round_winner_team': row.get('round_winner_team'),
        'team1_before': parse_int(row.get('team1_before')),
        'team2_before': parse_int(row.get('team2_before')),
        'team1_after': parse_int(row.get('team1_after')),
        'team2_after': parse_int(row.get('team2_after')),
        'trump': encode_suit(round_actions_parsed[0].get('trump')) if round_actions_parsed else None,
        'lead_suit': first_non_null_suit(round_actions_parsed),
    }

    for index in range(4):
        play = cards_played_parsed[index] if len(cards_played_parsed) > index else (None, None)
        action = get_round_action(round_actions_parsed, index)

        cleaned_row[f'play_{index + 1}_position'] = play[0]
        cleaned_row[f'play_{index + 1}_card'] = play[1]
        cleaned_row[f'cards_in_trick_{index + 1}'] = action_to_bits(action, 'cards_in_trick')
        cleaned_row[f'hand_before_play_{index + 1}'] = action_to_bits(action, 'hand_before')
        cleaned_row[f'legal_moves_play_{index + 1}'] = action_to_bits(action, 'legal_moves')

    return cleaned_row

def ordered_output_columns():
    ordered_columns = list(output_base_columns)
    ordered_columns.append('trump')

    for index in range(4):
        ordered_columns.extend([
            f'play_{index + 1}_position',
            f'play_{index + 1}_card',
            f'cards_in_trick_{index + 1}',
        ])
        if index == 0:
            ordered_columns.append('lead_suit')
        ordered_columns.extend([
            f'hand_before_play_{index + 1}',
            f'legal_moves_play_{index + 1}',
        ])

    return ordered_columns

def serialize_value(value):
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    if value is None:
        return ''
    return value

def clean_dataset(data_path, output_path=None):
    with open(data_path, 'r', encoding='utf-8-sig', newline='') as input_file:
        reader = csv.DictReader(input_file)
        cleaned_rows = [build_clean_row(row) for row in reader]

    if output_path is None:
        output_path = str(Path(data_path).with_name(f'{Path(data_path).stem}_cleaned.csv'))

    fieldnames = ordered_output_columns()
    with open(output_path, 'w', encoding='utf-8', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in cleaned_rows:
            writer.writerow({column: serialize_value(row.get(column)) for column in fieldnames})

    return output_path

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        Tk().withdraw()
        data_path = askopenfilename(filetypes=[("Ficheiros CSV", "*.csv"), ("Todos os ficheiros", "*.*")])

    if data_path:
        clean_dataset(data_path)
