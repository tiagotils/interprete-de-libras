"""
Sintetiza a trilha e os efeitos sonoros do Quiz do Setembro Azul.
Estilo Kahoot: pulso rítmico constante, arpejo de sintetizador, tensão leve.
Tudo gerado do zero com numpy — sem samples de terceiros, sem direito autoral.
"""
import numpy as np
from scipy.io import wavfile
import os

SR = 44100
SAIDA = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- utilidades
def nota(nome):
    """Converte nome de nota (ex: 'A4', 'C#5') em frequência."""
    passos = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    letra = nome[0].upper()
    resto = nome[1:]
    semi = passos[letra]
    if resto.startswith('#'):
        semi += 1
        resto = resto[1:]
    elif resto.startswith('b'):
        semi -= 1
        resto = resto[1:]
    oitava = int(resto)
    midi = 12 * (oitava + 1) + semi
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def env(n, ataque=0.01, decai=0.1, sustenta=0.7, solta=0.2):
    """Envelope ADSR em amostras."""
    a = max(1, int(ataque * SR))
    d = max(1, int(decai * SR))
    r = max(1, int(solta * SR))
    s = max(1, n - a - d - r)
    return np.concatenate([
        np.linspace(0, 1, a),
        np.linspace(1, sustenta, d),
        np.full(s, sustenta),
        np.linspace(sustenta, 0, r),
    ])[:n]


def onda(freq, dur, tipo='seno', detune=0.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    f = freq * (1 + detune)
    if tipo == 'seno':
        return np.sin(2 * np.pi * f * t)
    if tipo == 'triangulo':
        return 2 * np.abs(2 * ((f * t) % 1) - 1) - 1
    if tipo == 'serra':
        return 2 * ((f * t) % 1) - 1
    if tipo == 'quadrada':
        return np.sign(np.sin(2 * np.pi * f * t))
    raise ValueError(tipo)


def sintetiza(freq, dur, tipo='triangulo', vol=0.3, **adsr):
    x = onda(freq, dur, tipo)
    # leve engrossamento com uma segunda voz desafinada
    x = 0.7 * x + 0.3 * onda(freq, dur, tipo, detune=0.006)
    return x * env(len(x), **adsr) * vol


def kick(dur=0.18, vol=0.5):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    f = 110 * np.exp(-t * 28) + 42          # varredura descendente
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return x * np.exp(-t * 13) * vol


def chimbal(dur=0.055, vol=0.16):
    n = int(SR * dur)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    # filtro passa-alta simples por diferença
    x = np.diff(np.concatenate([[0], x]))
    return x * np.exp(-np.linspace(0, 1, n) * 22) * vol


def soma_em(base, som, inicio_s):
    i = int(inicio_s * SR)
    fim = min(len(base), i + len(som))
    if i < len(base):
        base[i:fim] += som[:fim - i]
    return base


def normaliza(x, pico=0.85):
    m = np.max(np.abs(x))
    return x * (pico / m) if m > 0 else x


def salva(nome, x, sr=SR):
    caminho = os.path.join(SAIDA, nome)
    wavfile.write(caminho, sr, (normaliza(x) * 32767).astype(np.int16))
    return caminho


# ------------------------------------------------------- trilha de fundo
def trilha():
    """
    Loop de 16 segundos, 120 BPM, 8 compassos.
    Progressão em lá menor: Am - F - C - G (a clássica dos quizzes).
    Emenda sem costura: o fim liga direto no começo.
    """
    bpm = 120
    tempo = 60.0 / bpm              # 0,5 s por tempo
    compasso = 4 * tempo            # 2 s
    total = 8 * compasso            # 16 s
    n = int(SR * total)
    mix = np.zeros(n)

    acordes = [
        (['A3', 'C4', 'E4'], 'A2'),
        (['F3', 'A3', 'C4'], 'F2'),
        (['C4', 'E4', 'G4'], 'C3'),
        (['G3', 'B3', 'D4'], 'G2'),
    ]

    for c in range(8):
        base = c * compasso
        notas, baixo = acordes[c % 4]

        # baixo pulsante em cada tempo
        for b in range(4):
            mix = soma_em(mix, sintetiza(nota(baixo), tempo * 0.9, 'triangulo',
                                         vol=0.30, ataque=0.005, decai=0.06,
                                         sustenta=0.5, solta=0.16),
                          base + b * tempo)

        # arpejo de colcheias subindo e descendo
        seq = [0, 1, 2, 1, 0, 1, 2, 1]
        for i, grau in enumerate(seq):
            mix = soma_em(mix, sintetiza(nota(notas[grau]) * 2, tempo / 2 * 0.85,
                                         'seno', vol=0.13, ataque=0.004,
                                         decai=0.05, sustenta=0.35, solta=0.1),
                          base + i * (tempo / 2))

        # colchão de acorde ao fundo
        for nt in notas:
            mix = soma_em(mix, sintetiza(nota(nt), compasso * 0.95, 'triangulo',
                                         vol=0.075, ataque=0.12, decai=0.3,
                                         sustenta=0.55, solta=0.5),
                          base)

        # percussão
        mix = soma_em(mix, kick(vol=0.42), base)
        mix = soma_em(mix, kick(vol=0.34), base + 2 * tempo)
        for b in range(8):
            if b % 2 == 1:
                mix = soma_em(mix, chimbal(), base + b * (tempo / 2))

    return mix


# ------------------------------------------------------------ efeitos
def efeito_acerto():
    """Arpejo ascendente de quinta — sensação de subida."""
    mix = np.zeros(int(SR * 0.85))
    for i, nt in enumerate(['C5', 'E5', 'G5', 'C6']):
        mix = soma_em(mix, sintetiza(nota(nt), 0.45, 'triangulo', vol=0.42,
                                     ataque=0.004, decai=0.09, sustenta=0.45,
                                     solta=0.3),
                      i * 0.075)
    # brilho por cima
    mix = soma_em(mix, sintetiza(nota('E6'), 0.4, 'seno', vol=0.16,
                                 ataque=0.01, decai=0.12, sustenta=0.3, solta=0.26),
                  0.3)
    return mix


def efeito_erro():
    """Duas notas descendentes, suaves — sem soar punitivo."""
    mix = np.zeros(int(SR * 0.7))
    mix = soma_em(mix, sintetiza(nota('E4'), 0.28, 'triangulo', vol=0.36,
                                 ataque=0.006, decai=0.08, sustenta=0.5, solta=0.15), 0.0)
    mix = soma_em(mix, sintetiza(nota('Bb3'), 0.42, 'triangulo', vol=0.36,
                                 ataque=0.006, decai=0.1, sustenta=0.45, solta=0.26), 0.17)
    return mix


def efeito_tique():
    """Tique curto do cronômetro nos últimos segundos."""
    return sintetiza(nota('A5'), 0.07, 'seno', vol=0.3,
                     ataque=0.002, decai=0.02, sustenta=0.25, solta=0.045)


def efeito_contagem():
    """Bipe da contagem regressiva antes da pergunta."""
    return sintetiza(nota('D5'), 0.16, 'triangulo', vol=0.38,
                     ataque=0.004, decai=0.04, sustenta=0.4, solta=0.1)


def efeito_comecar():
    """Fanfarra curta ao iniciar."""
    mix = np.zeros(int(SR * 0.7))
    for i, nt in enumerate(['G4', 'C5', 'E5']):
        mix = soma_em(mix, sintetiza(nota(nt), 0.4, 'triangulo', vol=0.4,
                                     ataque=0.005, decai=0.08, sustenta=0.5, solta=0.24),
                      i * 0.09)
    return mix


def efeito_fim():
    """Encerramento: acorde maior com resolução."""
    mix = np.zeros(int(SR * 1.9))
    fases = [
        (['C5', 'E5', 'G5'], 0.0, 0.5),
        (['D5', 'F5', 'A5'], 0.3, 0.5),
        (['E5', 'G5', 'C6'], 0.6, 1.2),
    ]
    for notas, inicio, dur in fases:
        for nt in notas:
            mix = soma_em(mix, sintetiza(nota(nt), dur, 'triangulo', vol=0.26,
                                         ataque=0.01, decai=0.15, sustenta=0.5,
                                         solta=dur * 0.5),
                          inicio)
    return mix


# --------------------------------------------------------------- execução
if __name__ == '__main__':
    print('gerando trilha...')
    salva('trilha.wav', trilha())

    for nome, fn in [
        ('acerto', efeito_acerto),
        ('erro', efeito_erro),
        ('tique', efeito_tique),
        ('contagem', efeito_contagem),
        ('comecar', efeito_comecar),
        ('fim', efeito_fim),
    ]:
        print('gerando', nome)
        salva(nome + '.wav', fn())

    print('pronto')
