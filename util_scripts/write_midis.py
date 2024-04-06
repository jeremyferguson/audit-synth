# from music21 import harmony

# import pretty_midi
import os

#adding unsupported chords to the harmony dictionary. probably a very unstable way to do this.
#harmony.CHORD_TYPES['minor fourth'] = ['1,-3,4',['m4']]
#harmony.CHORD_TYPES['major fourth'] = ['1,3,4',['M4']]
#harmony.CHORD_TYPES['diminished sixth'] = ['1,-3,-5,6',['dim6']]
def parseChord(chord):
    def parseRootNote(chord):
        match chord[1]:
            case 'b':
                return chord[0] + '-', chord[2:]
            case '#':
                return chord[0] + '#', chord[2:]
            case _:
                return chord[0], chord[1:]
    def parseQuality(chord):
        if chord[0] == '_':
            chord = chord[1:]
        match chord[0]:
            case 'm':
                return chord
            case 'M':
                if chord[1:] == '6':
                    return '6'
                return chord
            case 'd':
                return 'dim' + chord[1:]
            case _:
                raise ValueError(f"Unhandled chord: {chord}")

    rootNote, rest = parseRootNote(chord)
    quality = parseQuality(rest)
    #print(rootNote+quality)
    h = harmony.ChordSymbol(rootNote+quality)
    h.writeAsChord=True
    return h


source_dir = "/home/jmfergie/music/output"
def create_all_audio():
    for fname in os.listdir(source_dir):
        with open(os.path.join(source_dir,fname),'r') as f:
            text = f.read()
        chords = text.split(',')
        parsedChords = [parseChord(chord) for chord in chords]
        create_midi(parsedChords,fname.removesuffix('.txt'),'midifiles')

def create_midi(chords,fname,dir):
    parsedChords = [parseChord(chord) for chord in chords]
    midi_data = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    length = 1
    for n, chord in enumerate(parsedChords):
        for pitch in chord.pitches:
            note_number = pitch.midi
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=n * length, end=(n + 1) * length)
            piano.notes.append(note)
    midi_data.instruments.append(piano)
    midi_data.write(os.path.join(dir,f'{fname}.mid'))
