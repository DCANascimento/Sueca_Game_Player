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

def serialize_value(value):
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    if value is None:
        return ''
    return value

# --- NEW FUNCTION: Splits 1 round into up to 4 individual turns ---
def build_clean_turns(row):
    cards_played_parsed = parse_cards_played(row.get('cards_played'))
    round_actions_parsed = parse_round_actions(row.get('round_actions'))

    trump = encode_suit(round_actions_parsed[0].get('trump')) if round_actions_parsed else None
    lead_suit = first_non_null_suit(round_actions_parsed)

    turn_rows = []
    cards_on_table = [0] * 40  # Tracks what has already been played in this trick

    # Loop through each player's action inside the round
    for index in range(len(round_actions_parsed)):
        action = get_round_action(round_actions_parsed, index)
        play = cards_played_parsed[index] if len(cards_played_parsed) > index else (None, None)

        # Skip if there's missing action or target data
        if not action or play[1] is None:
            continue

        turn_row = {
            'game_number': parse_int(row.get('game_number')),
            'round_number': parse_int(row.get('round_number')),
            'round_points': parse_int(row.get('round_points')),
            'team1_before': parse_int(row.get('team1_before')),
            'team2_before': parse_int(row.get('team2_before')),
            'trump': trump,
            'lead_suit': lead_suit,
            'player_position': play[0],
            'position_in_trick': action.get('position_in_trick'),
            'cards_on_table_snapshot': list(cards_on_table), # Cards on table BEFORE this play
            'hand_before_play': action_to_bits(action, 'hand_before'),
            'legal_moves_play': action_to_bits(action, 'legal_moves'),
            'TARGET_card_played': play[1] # The label we want to predict
        }
        
        turn_rows.append(turn_row)

        # Add this played card to the table tracker so the NEXT player sees it
        played_card_id = play[1]
        if 0 <= played_card_id < 40:
            cards_on_table[played_card_id] = 1

    return turn_rows

# --- UPDATED FUNCTION: Handles writing the flat turns into the final CSV ---
def clean_dataset(data_path, output_path=None):
    all_flattened_turns = []

    with open(data_path, 'r', encoding='utf-8-sig', newline='') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            # We use .extend() because build_clean_turns returns a list of up to 4 items
            all_flattened_turns.extend(build_clean_turns(row))

    if output_path is None:
        output_path = str(Path(data_path).with_name(f'{Path(data_path).stem}_flat_turns.csv'))

    # Clean, flat column headers for Machine Learning
    fieldnames = [
        'game_number', 'round_number', 'round_points', 'team1_before', 'team2_before',
        'trump', 'lead_suit', 'player_position', 'position_in_trick',
        'cards_on_table_snapshot', 'hand_before_play', 'legal_moves_play', 'TARGET_card_played'
    ]

    with open(output_path, 'w', encoding='utf-8', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_flattened_turns:
            writer.writerow({column: serialize_value(row.get(column)) for column in fieldnames})

    print(f"Data conversion complete! Output saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        Tk().withdraw()
        data_path = askopenfilename(filetypes=[("Ficheiros CSV", "*.csv"), ("Todos os ficheiros", "*.*")])

    if data_path:
        clean_dataset(data_path)