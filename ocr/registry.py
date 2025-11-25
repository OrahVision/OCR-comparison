"""
OCR Provider Registry

Manages registration and retrieval of OCR providers.
"""

from typing import Dict, List, Type, Optional, Any
import logging

from .base import OCRProvider, ProviderType


class ProviderRegistry:
    """
    Registry for managing OCR providers.

    Provides centralized access to all registered OCR providers,
    with lazy instantiation and caching.
    """

    _providers: Dict[str, Type[OCRProvider]] = {}
    _instances: Dict[str, OCRProvider] = {}
    _configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, provider_class: Type[OCRProvider]) -> Type[OCRProvider]:
        """
        Register a provider class.

        Can be used as a decorator:
            @ProviderRegistry.register
            class MyOCRProvider(OCRProvider):
                ...

        Args:
            provider_class: The provider class to register

        Returns:
            The provider class (for decorator usage)
        """
        name = provider_class.PROVIDER_NAME
        cls._providers[name] = provider_class
        logging.debug(f"Registered OCR provider: {name}")
        return provider_class

    @classmethod
    def get(cls, name: str, **config) -> Optional[OCRProvider]:
        """
        Get a provider instance by name.

        Creates and caches instances on first access.

        Args:
            name: Provider name
            **config: Configuration to pass to provider constructor

        Returns:
            Provider instance or None if not found
        """
        if name not in cls._providers:
            logging.warning(f"Provider not found: {name}")
            return None

        # Check if we need to create a new instance
        config_key = str(config)
        cache_key = f"{name}:{config_key}"

        if cache_key not in cls._instances:
            try:
                cls._instances[cache_key] = cls._providers[name](**config)
                cls._configs[cache_key] = config
            except Exception as e:
                logging.error(f"Failed to instantiate provider {name}: {e}")
                return None

        return cls._instances[cache_key]

    @classmethod
    def get_class(cls, name: str) -> Optional[Type[OCRProvider]]:
        """
        Get a provider class by name (without instantiating).

        Args:
            name: Provider name

        Returns:
            Provider class or None if not found
        """
        return cls._providers.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List providers that are available (configured and ready).

        Returns:
            List of available provider names
        """
        available = []
        for name in cls._providers:
            try:
                provider = cls.get(name)
                if provider and provider.is_available():
                    available.append(name)
            except Exception:
                pass
        return available

    @classmethod
    def list_by_type(cls, provider_type: ProviderType) -> List[str]:
        """
        List providers of a specific type.

        Args:
            provider_type: Type filter (LOCAL, CLOUD, or LLM)

        Returns:
            List of provider names matching the type
        """
        return [
            name for name, provider_class in cls._providers.items()
            if provider_class.PROVIDER_TYPE == provider_type
        ]

    @classmethod
    def list_commercial(cls) -> List[str]:
        """
        List providers approved for commercial use.

        Returns:
            List of commercially-licensed provider names
        """
        return [
            name for name, provider_class in cls._providers.items()
            if provider_class.COMMERCIAL_USE
        ]

    @classmethod
    def get_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a provider.

        Args:
            name: Provider name

        Returns:
            Dictionary with provider info or None
        """
        if name not in cls._providers:
            return None

        provider_class = cls._providers[name]
        return {
            "name": provider_class.PROVIDER_NAME,
            "type": provider_class.PROVIDER_TYPE.value,
            "commercial_use": provider_class.COMMERCIAL_USE,
            "license": provider_class.LICENSE,
            "languages": provider_class.SUPPORTED_LANGUAGES,
        }

    @classmethod
    def clear_instances(cls):
        """Clear all cached provider instances."""
        cls._instances.clear()
        cls._configs.clear()

    @classmethod
    def clear_all(cls):
        """Clear all registered providers and instances."""
        cls._providers.clear()
        cls._instances.clear()
        cls._configs.clear()


def register_provider(provider_class: Type[OCRProvider]) -> Type[OCRProvider]:
    """
    Convenience function to register a provider.

    Usage:
        @register_provider
        class MyProvider(OCRProvider):
            ...
    """
    return ProviderRegistry.register(provider_class)


# Exports
__all__ = [
    'ProviderRegistry',
    'register_provider'
]
