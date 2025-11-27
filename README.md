# Repositorio-TCC

Este repositório contém o código-fonte do framework de detector de deepfake (DeepfakeBench) e o gerador de ataque adversario desenvolvido pelo autor, ambos utilizados no artigo.

## Estrutura do Projeto

O projeto foi estruturado em dois módulos independentes para evitar conflitos de dependências:

### 1. [Framework de Detecção](./DeepfakeBench)
Contém a estrutura base do framework **DeepfakeBench**, utilizado para o teste dos modelos de detecção (MesoNetIncept e CNNaug).
* **Origem:** Baseado no projeto [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).
* **Função:** Classificar as imagens como Reais ou Falsas.

### 2. [Gerador de Ataques](./AtaqueAdversario)
Módulo desenvolvido especificamente para este trabalho.
* **Autoria:** Própria.
* **Função:** Aplica ruído adversário (Fast Gradient Sign Method) nas imagens do dataset para gerar os casos de teste.

---

## Instalação e Ambientes

Devido a diferenças nas bibliotecas, recomenda-se o uso de ambientes virtuais separados:

* **Para rodar o Ataque:** Instale as dependências listadas em `AtaqueAdversario/requirements.txt`.
* **Para rodar a Detecção:** Utilize o script `install.sh` dentro da pasta `DeepfakeBench`. (Caso queira reproduzir exatamente, foi utilizado o condalabs no googlecolab)

---

## Datasets e Pesos

Os Datasets originais, backbones pre-treinados e pesos pre-treinados podem ser encontrados no repositorio do [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) nas seções 2, 5 e 6 respectivamente.

---

## Licença e Créditos

Este projeto é uma obra derivada e acadêmica, distribuída sob a licença **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

**Créditos do Projeto Base:**
O módulo de detecção utiliza código e modelos pré-treinados do projeto **DeepfakeBench**, desenvolvido por *The Chinese University of Hong Kong, Shenzhen*.

> *Copyright (c) 2023, CUHK(SZ). All rights reserved.*

O uso deste software é restrito a fins acadêmicos e não comerciais. Para mais detalhes sobre os termos de uso e licenças de terceiros, consulte o arquivo [LICENSE](./LICENSE) na raiz deste repositório.
