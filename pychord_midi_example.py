#source: https://github.com/yuma-m/pychord/blob/main/examples/pychord-midi.py
import pretty_midi

from pychord import Chord


def create_midi(chords):
    midi_data = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    length = 1
    for n, chord in enumerate(chords):
        print(chord)
        for note_name in chord.components_with_pitch(root_pitch=4):
            print(note_name)
            note_number = pretty_midi.note_name_to_number(note_name)
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=n * length, end=(n + 1) * length)
            piano.notes.append(note)
    midi_data.instruments.append(piano)
    midi_data.write('chord.mid')


def main():
    chords_str = ["C", "Dm7", "G", "C"]
    chords = [Chord(c) for c in chords_str]
    create_midi(chords)


if __name__ == '__main__':
    main()