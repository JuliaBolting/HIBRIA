"""HTTP público com DNS fixado por conexão, TLS verificado e limites de leitura.

Não usa proxies do ambiente nem segue redirects do urllib3. Cada salto passa
pela mesma validação. Ollama é uma conexão administrativa separada e local.
"""
from __future__ import annotations

import ipaddress
import socket
import time
from urllib.parse import urljoin, urlsplit

import requests as _requests
import urllib3
from requests import RequestException, Response, exceptions, utils

MAX_BYTES = 6 * 1024 * 1024
MAX_REDIRECTS = 5


class UnsafeURL(RequestException, ValueError):
    pass


def validate_url(url: str, *, resolve: bool = True) -> tuple[str, int, list[str]]:
    try:
        parts = urlsplit(str(url))
        host = (parts.hostname or "").encode("idna").decode("ascii").lower().rstrip(".")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if (parts.scheme not in {"http", "https"} or not host or parts.username is not None
                or parts.password is not None or port not in {80, 443}
                or any(ord(c) < 33 for c in str(url)) or "\\" in str(url)
                or host == "localhost" or host.endswith((".localhost", ".local", ".internal"))):
            raise ValueError
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None and not _public_ip(str(literal)):
            raise ValueError
        addresses = ([str(literal)] if literal else
                     list(dict.fromkeys(item[4][0] for item in socket.getaddrinfo(
                         host, port, type=socket.SOCK_STREAM))) if resolve else [])
        if resolve and (not addresses or not all(_public_ip(ip) for ip in addresses)):
            raise ValueError
        return host, port, addresses
    except (ValueError, UnicodeError, OSError):
        raise UnsafeURL("URL indisponível ou não permitida para acesso público.") from None


def _public_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped or ip.sixtofour or ip.teredo or ip in ipaddress.ip_network("64:ff9b::/96"):
            return False
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_unspecified)


class PublicResponse(Response):
    def raise_for_status(self):
        if self.status_code >= 400:
            raise exceptions.HTTPError(f"HTTP {self.status_code} no serviço externo.", response=self)


def request(method, url, *, params=None, headers=None, json=None, data=None,
            timeout=15, allow_redirects=True, stream=False, **kwargs):
    if kwargs:
        raise TypeError("Opções HTTP não suportadas: " + ", ".join(sorted(kwargs)))
    prepared = _requests.Request(method, str(url), params=params, headers=headers,
                                 json=json, data=data).prepare()
    current = prepared.url
    method, body = prepared.method, prepared.body
    outgoing = dict(prepared.headers)
    outgoing = {k:v for k,v in outgoing.items() if k.lower() != "host"}
    outgoing["Accept-Encoding"] = "gzip, deflate"
    if isinstance(body, str):
        body = body.encode("utf-8")
    connect, read = timeout if isinstance(timeout, tuple) else (timeout, timeout)
    connect, read = min(float(connect or 10), 15), min(float(read or 30), 60)
    deadline = time.monotonic() + 60
    history = []
    for hop in range(MAX_REDIRECTS + 1):
        host, port, addresses = validate_url(current)
        parts = urlsplit(current)
        # O pool recebe o IP validado, nunca o hostname a ser resolvido de novo.
        pool_options = {"host": addresses[0], "port": port, "maxsize": 1}
        if parts.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(**pool_options, server_hostname=host,
                assert_hostname=host, cert_reqs="CERT_REQUIRED", ca_certs=_requests.certs.where())
        else:
            pool = urllib3.HTTPConnectionPool(**pool_options)
        netloc = f"[{host}]" if ":" in host else host
        outgoing["Host"] = netloc if port == (443 if parts.scheme == "https" else 80) else f"{netloc}:{port}"
        raw = None
        try:
            raw = pool.urlopen(method, parts.path or "/" if not parts.query else
                               (parts.path or "/") + "?" + parts.query,
                               body=body, headers=outgoing, redirect=False, retries=False,
                               preload_content=False, assert_same_host=False,
                               timeout=urllib3.Timeout(connect=connect, read=read))
            response = PublicResponse()
            response.status_code = raw.status
            response.headers = _requests.structures.CaseInsensitiveDict(raw.headers)
            response.url = current
            response.history = list(history)
            response.encoding = utils.get_encoding_from_headers(response.headers)
            response._content = b""
            response._content_consumed = True
            location = response.headers.get("Location")
            if allow_redirects and response.status_code in {301, 302, 303, 307, 308} and location:
                if hop == MAX_REDIRECTS:
                    raise RequestException("Excesso de redirecionamentos.")
                next_url = urljoin(current, location)
                next_parts = urlsplit(next_url)
                if (next_parts.scheme, next_parts.netloc) != (parts.scheme, parts.netloc):
                    if method not in {"GET", "HEAD"}:
                        raise UnsafeURL("Redirecionamento de credenciais não permitido.")
                    outgoing = {k:v for k,v in outgoing.items() if k.lower() in {
                        "user-agent", "accept", "accept-language", "accept-encoding"}}
                if response.status_code == 303 and method != "HEAD":
                    method, body = "GET", None
                history.append(response)
                current = next_url
                continue
            chunks, size = [], 0
            for chunk in raw.stream(65536, decode_content=True):
                size += len(chunk)
                if size > MAX_BYTES or time.monotonic() > deadline:
                    raise RequestException("Resposta externa excedeu o limite de tamanho ou tempo.")
                chunks.append(chunk)
            response._content = b"".join(chunks)
            return response
        except RequestException:
            raise
        except urllib3.exceptions.TimeoutError:
            raise exceptions.Timeout("Tempo de acesso ao serviço externo esgotado.") from None
        except (urllib3.exceptions.HTTPError, OSError):
            raise exceptions.ConnectionError("Não foi possível acessar o serviço externo.") from None
        finally:
            if raw is not None:
                raw.close()
            pool.close()
    raise RequestException("Redirecionamento inválido.")


def get(url, **kwargs):
    return request("GET", url, **kwargs)


def post(url, **kwargs):
    return request("POST", url, **kwargs)
