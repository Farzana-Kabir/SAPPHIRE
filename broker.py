import pika, logging, threading, msgpack, ssl, os
from typing import Callable, Dict
from dotenv import load_dotenv

for _candidate in ("_env", ".env"):
    if os.path.exists(_candidate):
        load_dotenv(_candidate)
        break

class FLMessageBroker:
    @staticmethod
    def _with_query_defaults(url, heartbeat, blocked_connection_timeout):
        from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        query.setdefault("heartbeat", str(heartbeat))
        query.setdefault("blocked_connection_timeout", str(blocked_connection_timeout))
        return urlunsplit(parts._replace(query=urlencode(query)))

    def __init__(self, host="localhost", ca_cert_path="ca_certificate.pem",
                 queue_name="fl_server_queue_v2", exchange_name="fl_broadcast_exchange",
                 heartbeat=3600, blocked_connection_timeout=3600):
        self.queue_name = queue_name
        self.exchange_name = exchange_name

        if host == "localhost":
            host = os.getenv("RABBITMQ_HOST") or os.getenv("RABBITMQ_URL") or "localhost"

        if host.startswith("amqp://") or host.startswith("amqps://"):
            self.parameters = pika.URLParameters(
                self._with_query_defaults(host, heartbeat, blocked_connection_timeout)
            )
        else:
            self.credentials = pika.PlainCredentials('guest', 'guest')
            context = ssl.create_default_context(cafile=ca_cert_path) if os.path.exists(ca_cert_path) else ssl.create_default_context()
            context.check_hostname = False
            ssl_options = pika.SSLOptions(context, host)
            self.parameters = pika.ConnectionParameters(
                host=host, port=5671, credentials=self.credentials,
                ssl_options=ssl_options, heartbeat=heartbeat,
                blocked_connection_timeout=blocked_connection_timeout
            )
            
        self._send_lock = threading.Lock()
        self._send_conn = None
        self._send_channel = None

    def _get_connection(self):
        return pika.BlockingConnection(self.parameters)

    def _setup_topology(self, channel):
        queue_args = {"x-max-length": 10000, "x-max-length-bytes": 52428800, "x-overflow": "drop-head"}
        channel.queue_declare(queue=self.queue_name, durable=True, arguments=queue_args)
        channel.exchange_declare(exchange=self.exchange_name, exchange_type='fanout', durable=True)

    def _ensure_sender_ready(self):
        if not self._send_conn or self._send_conn.is_closed:
            self._send_conn = self._get_connection()
            self._send_channel = self._send_conn.channel()
            self._setup_topology(self._send_channel)
        elif self._send_channel.is_closed:
            self._send_channel = self._send_conn.channel()
            self._setup_topology(self._send_channel)

    def _safe_close_sender(self):
        try:
            if self._send_conn and not self._send_conn.is_closed:
                self._send_conn.close()
        except Exception:
            pass

    def listen_client_queues(self, callback: Callable):
        connection = self._get_connection()
        channel = connection.channel()
        self._setup_topology(channel)

        def wrapper(ch, method, properties, body):
            try:
                msg = msgpack.unpackb(body, strict_map_key=False)
            except Exception as e:
                logging.error(f"Malformed message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return
            try:
                callback(msg)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logging.error(f"Callback exception: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.queue_name, on_message_callback=wrapper)
        try:
            channel.start_consuming()
        except Exception as e:
            logging.error(f"Listener stopped: {e}")
        finally:
            if not connection.is_closed:
                connection.close()

    def listen_broadcasts(self, callback: Callable):
        connection = self._get_connection()
        channel = connection.channel()
        self._setup_topology(channel)

        result = channel.queue_declare(queue='', exclusive=True)
        client_queue = result.method.queue
        channel.queue_bind(exchange=self.exchange_name, queue=client_queue)

        def wrapper(ch, method, properties, body):
            try:
                msg = msgpack.unpackb(body, strict_map_key=False)
            except Exception as e:
                logging.error(f"Malformed broadcast: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return
            try:
                callback(msg)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logging.error(f"Broadcast callback exception: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        channel.basic_consume(queue=client_queue, on_message_callback=wrapper)
        try:
            channel.start_consuming()
        except Exception as e:
            logging.error(f"Broadcast listener stopped: {e}")
        finally:
            if not connection.is_closed:
                connection.close()

    def broadcast(self, msg: Dict):
        with self._send_lock:
            self._ensure_sender_ready()
            try:
                self._send_channel.basic_publish(
                    exchange=self.exchange_name, routing_key='', body=msgpack.packb(msg, use_bin_type=True)
                )
            except Exception as e:
                logging.error(f"Broadcast failed: {e}")
                self._safe_close_sender()       
                self._ensure_sender_ready()
                self._send_channel.basic_publish(
                    exchange=self.exchange_name, routing_key='', body=msgpack.packb(msg, use_bin_type=True)
                )

    def send_to_server(self, msg: Dict):
        with self._send_lock:
            self._ensure_sender_ready()
            try:
                self._send_channel.basic_publish(
                    exchange='', routing_key=self.queue_name, body=msgpack.packb(msg, use_bin_type=True)
                )
            except Exception as e:
                logging.error(f"Send failed: {e}")
                self._safe_close_sender()       
                self._ensure_sender_ready()
                self._send_channel.basic_publish(
                    exchange='', routing_key=self.queue_name, body=msgpack.packb(msg, use_bin_type=True)
                )
