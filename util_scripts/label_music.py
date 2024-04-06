import os
from tkinter import Tk, Label
#import pygame


#pygame.init()
#pygame.mixer.init(frequency=44100, size=-16, channels=1)

class SongSorter:
    def __init__(self, root):
        self.root = root
        self.song_dir = "/home/jmfergie/music/output"
        self.yes_dir = "/home/jmfergie/music/liked"
        self.no_dir = "/home/jmfergie/music/disliked"
        self.midi_dir = "/home/jmfergie/audit-synth/midifiles"
        self.current_song_idx = 0
        self.songs = self.get_song_files()
        
        self.label = Label(root)
        self.label.pack()
        self.display_song()
        self.root.bind('p', self.play_current_song)
        self.root.bind('y', self.move_to_yes)
        self.root.bind('n', self.move_to_no)

    def get_song_files(self):
        song_files = [f for f in os.listdir(self.song_dir) if f.endswith(('.txt'))]
        return song_files
    
    def play_current_song(self,_):
        if self.current_song_idx < len(self.songs):
            midi_path = os.path.join(self.midi_dir, self.songs[self.current_song_idx].removesuffix('.txt')+'.mid')
            pygame.mixer.music.load(midi_path,namehint="midi")
            pygame.mixer.music.play()
            

    def display_song(self):
        if self.current_song_idx < len(self.songs):
            song_path = os.path.join(self.song_dir, self.songs[self.current_song_idx])
            with open(song_path,'r') as f:
                text = f.read()
            chords = text.split(',')
            label_out = self.songs[self.current_song_idx] + '\n'
            for i in range(0,len(chords),10):
                label_out += ','.join(chords[i:i+10]) + '\n'
            self.label.config(text=label_out)
        else:
            self.label.config(text="No more songs to display")

    def move_to_yes(self, event=None):
        # Move the current song to the "yes" directory
        if self.current_song_idx < len(self.songs):
            current_song = self.songs[self.current_song_idx]
            source_path = os.path.join(self.song_dir, current_song)
            destination_path = os.path.join(self.yes_dir, current_song)
            os.rename(source_path, destination_path)
            self.current_song_idx += 1
            self.display_song()

    def move_to_no(self, event=None):
        # Move the current song to the "no" directory
        if self.current_song_idx < len(self.songs):
            current_song = self.songs[self.current_song_idx]
            source_path = os.path.join(self.song_dir, current_song)
            destination_path = os.path.join(self.no_dir, current_song)
            os.rename(source_path, destination_path)
            self.current_song_idx += 1
            self.display_song()

if __name__ == "__main__":
    root = Tk()
    root.title("Music Sorter")
    app = SongSorter(root)
    root.mainloop()
    #create_all_audio()


