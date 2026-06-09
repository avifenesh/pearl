package jsonrpc

import (
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
)

func init() {
	httpcaddyfile.RegisterHandlerDirective("jsonrpc", parseCaddyfile)
	httpcaddyfile.RegisterDirectiveOrder("jsonrpc", "before", "reverse_proxy")
}

// parseCaddyfile unmarshals tokens from h into a new Handler.
func parseCaddyfile(h httpcaddyfile.Helper) (caddyhttp.MiddlewareHandler, error) {
	var handler Handler
	err := handler.UnmarshalCaddyfile(h.Dispenser)
	return &handler, err
}

// UnmarshalCaddyfile implements caddyfile.Unmarshaler. Syntax:
//
//	jsonrpc {
//	    allow <method> [<method>...]
//	    cache <method> <ttl>
//	}
//
// allow may be repeated; methods accumulate across lines. Each argument may be
// a Caddy placeholder such as {env.RPC_ALLOWED_METHODS}, which is resolved (and
// split on whitespace/commas) in Provision. When no allow directive is present —
// or it resolves to nothing — every method is permitted.
func (h *Handler) UnmarshalCaddyfile(d *caddyfile.Dispenser) error {
	for d.Next() {
		if d.NextArg() {
			return d.ArgErr()
		}

		for nesting := d.Nesting(); d.NextBlock(nesting); {
			switch d.Val() {
			case "allow":
				args := d.RemainingArgs()
				if len(args) == 0 {
					return d.ArgErr()
				}
				h.Allow = append(h.Allow, args...)
			case "cache":
				args := d.RemainingArgs()
				if len(args) != 2 {
					return d.ArgErr()
				}
				ttl, err := time.ParseDuration(args[1])
				if err != nil {
					return d.Errf("invalid TTL %q: %v", args[1], err)
				}
				h.Rules = append(h.Rules, CacheRule{
					Method: args[0],
					TTL:    caddy.Duration(ttl),
				})
			default:
				return d.Errf("unrecognized subdirective '%s'", d.Val())
			}
		}
	}
	return nil
}

// Interface guard
var _ caddyfile.Unmarshaler = (*Handler)(nil)
