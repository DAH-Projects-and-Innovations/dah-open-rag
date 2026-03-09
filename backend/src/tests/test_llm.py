import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

# ÉTAPE 1 : Ajouter le dossier backend au path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ÉTAPE 2 : Charger les variables d'environnement
from dotenv import load_dotenv
load_dotenv()

# ÉTAPE 3 : Imports réels du projet
from src.core.models import Document
from src.core.interfaces import ILLM


def load_llm_class(filename, class_name, extra_mocks=None):
    """
    Charge une classe LLM depuis un fichier old/ en remplaçant
    les imports relatifs (..core) par les vrais modules du projet.

    Args:
        filename  : nom du fichier dans src/llm/old/ (ex: mistral_llm.py)
        class_name: nom de la classe à récupérer (ex: MistralLLM)
        extra_mocks: dict de mocks supplémentaires à injecter dans le module

    Returns:
        La classe demandée, chargée depuis le vrai fichier
    """
    filepath = os.path.join(BACKEND_DIR, "src", "llm", "old", filename)

    # Lire le code source du fichier
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    # Remplacer les imports relatifs par des imports absolus
    source = source.replace(
        "from ..core.interfaces import ILLM",
        "from src.core.interfaces import ILLM"
    )
    source = source.replace(
        "from ..core.models import Document",
        "from src.core.models import Document"
    )

    # Créer un nouveau module Python
    module_name = filename.replace(".py", "")
    module = types.ModuleType(module_name)
    module.__file__ = filepath

    # Injecter les mocks des librairies manquantes dans le module
    if extra_mocks:
        for key, value in extra_mocks.items():
            sys.modules[key] = value

    # Exécuter le code source dans ce module
    exec(compile(source, filepath, "exec"), module.__dict__)

    return getattr(module, class_name)


# Charger les 3 classes LLM

# Mistral :
MistralLLM = load_llm_class("mistral_llm.py", "MistralLLM")

# OpenAI :
mock_openai_lib = MagicMock()
sys.modules['openai'] = mock_openai_lib
OpenAILLM = load_llm_class("openai_llm.py", "OpenAILLM")

# LocalLLM :
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_community.llms'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.callbacks'] = MagicMock()
sys.modules['langchain_core.prompts'] = MagicMock()
LocalLLM = load_llm_class("ollama_llm.py", "LocalLLM", extra_mocks={
    'langchain_community.llms.LlamaCpp': MagicMock(),
})


# TESTS MISTRAL (vrais appels API)

class TestMistralLLM(unittest.TestCase):
    """Tests réels du MistralLLM avec l'API Mistral."""

    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise unittest.SkipTest(
                "MISTRAL_API_KEY non trouvée dans .env — tests Mistral ignorés."
            )
        cls.api_key = api_key
        cls.llm = MistralLLM(model_name="mistral-small-latest", api_key=api_key)
        print("\n✅ MistralLLM initialisé avec succès")

    def test_01_initialisation(self):
        """Vérifie que le client Mistral est bien initialisé."""
        self.assertIsNotNone(self.llm.client)
        self.assertEqual(self.llm.model_name, "mistral-small-latest")
        print("   ✅ Initialisation OK")

    def test_02_generate_simple(self):
        """Test d'une génération simple sans contexte."""
        print("\n   ⏳ Appel API Mistral (generate)...")
        response = self.llm.generate("Réponds en une phrase : qu'est-ce que Python ?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)
        print(f"   ✅ Réponse reçue : {response[:120]}...")

    def test_03_generate_with_context(self):
        """Test d'une génération RAG avec contexte de documents."""
        print("\n   ⏳ Appel API Mistral (generate_with_context)...")
        docs = [
            Document(content=(
                "Le RAG (Retrieval-Augmented Generation) est une technique qui combine "
                "la recherche de documents avec la génération de texte par un LLM."
            )),
            Document(content=(
                "Le RAG permet d'améliorer la précision des réponses en fournissant "
                "des documents pertinents au modèle."
            ))
        ]
        response = self.llm.generate_with_context(
            query="Qu'est-ce que le RAG ?",
            context=docs
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)
        print(f"   ✅ Réponse avec contexte : {response[:150]}...")

    def test_04_anti_hallucination(self):
        """Vérifie que le modèle refuse de répondre hors contexte."""
        print("\n   ⏳ Test anti-hallucination...")
        docs = [Document(content="Le ciel est bleu à cause de la diffusion de Rayleigh.")]
        response = self.llm.generate_with_context(
            query="Quel est le prix du Bitcoin aujourd'hui ?",
            context=docs
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 5)
        print(f"   ✅ Réponse hors contexte : {response[:150]}...")

    def test_05_prompt_personnalise(self):
        """Test avec un prompt système personnalisé."""
        print("\n   ⏳ Test prompt personnalisé...")
        llm_custom = MistralLLM(
            api_key=self.api_key,
            system_prompt="Tu es un assistant très concis. Réponds en maximum 1 phrase.",
        )
        response = llm_custom.generate("Qu'est-ce que l'intelligence artificielle ?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 5)
        print(f"   ✅ Réponse avec prompt custom : {response[:150]}...")


# TESTS OPENAI (mocks — pas de clé disponible)

class TestOpenAILLM(unittest.TestCase):
    """Tests du OpenAILLM avec mocks."""

    def setUp(self):
        self.llm = OpenAILLM(api_key="fake-key-test")
        # Remplacer le client par un vrai mock contrôlé
        self.llm.client = MagicMock()

    def test_01_initialisation(self):
        """Vérifie que le client OpenAI est bien initialisé."""
        self.assertIsNotNone(self.llm.client)
        self.assertEqual(self.llm.model_name, "gpt-4o-mini")
        print("\n   ✅ OpenAI initialisation OK (mock)")

    def test_02_generate_simple(self):
        """Test génération simple avec mock."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Python est un langage de programmation."
        self.llm.client.chat.completions.create.return_value = mock_response

        response = self.llm.generate("Qu'est-ce que Python ?")
        self.assertEqual(response, "Python est un langage de programmation.")
        self.llm.client.chat.completions.create.assert_called_once()
        print("   ✅ OpenAI generate OK (mock)")

    def test_03_generate_with_context(self):
        """Test génération RAG avec mock."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Le RAG combine recherche et génération."
        self.llm.client.chat.completions.create.return_value = mock_response

        docs = [Document(content="Le RAG est une technique d'IA.")]
        response = self.llm.generate_with_context("Qu'est-ce que le RAG ?", docs)
        self.assertEqual(response, "Le RAG combine recherche et génération.")
        print("   ✅ OpenAI generate_with_context OK (mock)")

    def test_04_prompt_systeme_par_defaut(self):
        """Vérifie que le prompt système contient les instructions anti-hallucination."""
        self.assertIn("EXCLUSIVEMENT", self.llm.default_system_prompt)
        self.assertIn("documents fournis", self.llm.default_system_prompt)
        print("   ✅ OpenAI prompt système anti-hallucination OK")

    def test_05_prompt_personnalise(self):
        """Test avec prompt personnalisé."""
        llm_custom = OpenAILLM(
            api_key="fake-key",
            system_prompt="Tu es un assistant concis.",
            user_prompt_template="Question: {query}\nContexte: {context_str}"
        )
        self.assertEqual(llm_custom.default_system_prompt, "Tu es un assistant concis.")
        print("   ✅ OpenAI prompt personnalisé OK")


# TESTS LOCAL LLM / OLLAMA (mocks — non installé)

class TestLocalLLM(unittest.TestCase):
    """Tests du LocalLLM (LlamaCpp) avec mocks."""

    @patch('os.path.exists', return_value=True)
    def test_01_initialisation(self, mock_exists):
        """Vérifie que LocalLLM s'initialise correctement."""
        llm = LocalLLM(model_path="models/test.gguf")
        self.assertEqual(llm.model_path, "models/test.gguf")
        print("\n   ✅ LocalLLM initialisation OK (mock)")

    @patch('os.path.exists', return_value=False)
    def test_02_fichier_introuvable(self, mock_exists):
        """Vérifie que LocalLLM lève une erreur si le fichier n'existe pas."""
        with self.assertRaises(FileNotFoundError):
            LocalLLM(model_path="models/inexistant.gguf")
        print("   ✅ LocalLLM FileNotFoundError OK")

    @patch('os.path.exists', return_value=True)
    def test_03_generate(self, mock_exists):
        """Test de génération avec mock."""
        llm = LocalLLM(model_path="models/test.gguf")
        llm.llm = MagicMock()
        llm.llm.invoke.return_value = "Réponse locale générée."
        response = llm.generate("Bonjour")
        self.assertEqual(response, "Réponse locale générée.")
        print("   ✅ LocalLLM generate OK (mock)")

    @patch('os.path.exists', return_value=True)
    def test_04_generate_with_context(self, mock_exists):
        """Test de génération RAG avec mock."""
        llm = LocalLLM(model_path="models/test.gguf")
        llm.llm = MagicMock()
        llm.llm.invoke.return_value = "Réponse RAG locale."
        docs = [Document(content="Contenu du document de test.")]
        response = llm.generate_with_context("Question test", docs)
        self.assertEqual(response, "Réponse RAG locale.")
        print("   ✅ LocalLLM generate_with_context OK (mock)")

    @patch('os.path.exists', return_value=True)
    def test_05_prompt_systeme_par_defaut(self, mock_exists):
        """Vérifie le prompt système anti-hallucination."""
        llm = LocalLLM(model_path="models/test.gguf")
        self.assertIn("Je ne dispose pas", llm.default_system_prompt)
        print("   ✅ LocalLLM prompt anti-hallucination OK")


# MAIN
if __name__ == "__main__":
    print("=" * 60)
    print(" TESTS LLM - RAG-DataAfriqueHub")
    print("   Branche : test-llm")
    print("=" * 60)
    print(" Légende :")
    print("   Mistral → vrais appels API (MISTRAL_API_KEY)")
    print("   OpenAI  → mocks (pas de clé)")
    print("   Local   → mocks (Ollama non installé)")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMistralLLM))
    suite.addTests(loader.loadTestsFromTestCase(TestOpenAILLM))
    suite.addTests(loader.loadTestsFromTestCase(TestLocalLLM))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    nb_success = result.testsRun - len(result.failures) - len(result.errors)
    print(f"✅ Tests réussis  : {nb_success}/{result.testsRun}")
    print(f"❌ Échecs         : {len(result.failures)}")
    print(f"💥 Erreurs        : {len(result.errors)}")
    print("=" * 60)