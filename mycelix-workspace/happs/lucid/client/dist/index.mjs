// node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs
var UINT32_MAX = 4294967295;
function setInt64(view, offset, value) {
  var high = Math.floor(value / 4294967296);
  var low = value;
  view.setUint32(offset, high);
  view.setUint32(offset + 4, low);
}
function getInt64(view, offset) {
  var high = view.getInt32(offset);
  var low = view.getUint32(offset + 4);
  return high * 4294967296 + low;
}
function getUint64(view, offset) {
  var high = view.getUint32(offset);
  var low = view.getUint32(offset + 4);
  return high * 4294967296 + low;
}

// node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs
var _a;
var _b;
var _c;
var TEXT_ENCODING_AVAILABLE = (typeof process === "undefined" || ((_a = process === null || process === void 0 ? void 0 : process.env) === null || _a === void 0 ? void 0 : _a["TEXT_ENCODING"]) !== "never") && typeof TextEncoder !== "undefined" && typeof TextDecoder !== "undefined";
var sharedTextEncoder = TEXT_ENCODING_AVAILABLE ? new TextEncoder() : void 0;
var TEXT_ENCODER_THRESHOLD = !TEXT_ENCODING_AVAILABLE ? UINT32_MAX : typeof process !== "undefined" && ((_b = process === null || process === void 0 ? void 0 : process.env) === null || _b === void 0 ? void 0 : _b["TEXT_ENCODING"]) !== "force" ? 200 : 0;
function utf8EncodeTEencode(str, output, outputOffset) {
  output.set(sharedTextEncoder.encode(str), outputOffset);
}
function utf8EncodeTEencodeInto(str, output, outputOffset) {
  sharedTextEncoder.encodeInto(str, output.subarray(outputOffset));
}
var utf8EncodeTE = (sharedTextEncoder === null || sharedTextEncoder === void 0 ? void 0 : sharedTextEncoder.encodeInto) ? utf8EncodeTEencodeInto : utf8EncodeTEencode;
var CHUNK_SIZE = 4096;
function utf8DecodeJs(bytes, inputOffset, byteLength) {
  var offset = inputOffset;
  var end = offset + byteLength;
  var units = [];
  var result = "";
  while (offset < end) {
    var byte1 = bytes[offset++];
    if ((byte1 & 128) === 0) {
      units.push(byte1);
    } else if ((byte1 & 224) === 192) {
      var byte2 = bytes[offset++] & 63;
      units.push((byte1 & 31) << 6 | byte2);
    } else if ((byte1 & 240) === 224) {
      var byte2 = bytes[offset++] & 63;
      var byte3 = bytes[offset++] & 63;
      units.push((byte1 & 31) << 12 | byte2 << 6 | byte3);
    } else if ((byte1 & 248) === 240) {
      var byte2 = bytes[offset++] & 63;
      var byte3 = bytes[offset++] & 63;
      var byte4 = bytes[offset++] & 63;
      var unit = (byte1 & 7) << 18 | byte2 << 12 | byte3 << 6 | byte4;
      if (unit > 65535) {
        unit -= 65536;
        units.push(unit >>> 10 & 1023 | 55296);
        unit = 56320 | unit & 1023;
      }
      units.push(unit);
    } else {
      units.push(byte1);
    }
    if (units.length >= CHUNK_SIZE) {
      result += String.fromCharCode.apply(String, units);
      units.length = 0;
    }
  }
  if (units.length > 0) {
    result += String.fromCharCode.apply(String, units);
  }
  return result;
}
var sharedTextDecoder = TEXT_ENCODING_AVAILABLE ? new TextDecoder() : null;
var TEXT_DECODER_THRESHOLD = !TEXT_ENCODING_AVAILABLE ? UINT32_MAX : typeof process !== "undefined" && ((_c = process === null || process === void 0 ? void 0 : process.env) === null || _c === void 0 ? void 0 : _c["TEXT_DECODER"]) !== "force" ? 200 : 0;
function utf8DecodeTD(bytes, inputOffset, byteLength) {
  var stringBytes = bytes.subarray(inputOffset, inputOffset + byteLength);
  return sharedTextDecoder.decode(stringBytes);
}

// node_modules/@msgpack/msgpack/dist.es5+esm/ExtData.mjs
var ExtData = (
  /** @class */
  /* @__PURE__ */ (function() {
    function ExtData2(type, data) {
      this.type = type;
      this.data = data;
    }
    return ExtData2;
  })()
);

// node_modules/@msgpack/msgpack/dist.es5+esm/DecodeError.mjs
var __extends = /* @__PURE__ */ (function() {
  var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(d2, b2) {
      d2.__proto__ = b2;
    } || function(d2, b2) {
      for (var p in b2) if (Object.prototype.hasOwnProperty.call(b2, p)) d2[p] = b2[p];
    };
    return extendStatics(d, b);
  };
  return function(d, b) {
    if (typeof b !== "function" && b !== null)
      throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
    extendStatics(d, b);
    function __() {
      this.constructor = d;
    }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
  };
})();
var DecodeError = (
  /** @class */
  (function(_super) {
    __extends(DecodeError2, _super);
    function DecodeError2(message) {
      var _this = _super.call(this, message) || this;
      var proto = Object.create(DecodeError2.prototype);
      Object.setPrototypeOf(_this, proto);
      Object.defineProperty(_this, "name", {
        configurable: true,
        enumerable: false,
        value: DecodeError2.name
      });
      return _this;
    }
    return DecodeError2;
  })(Error)
);

// node_modules/@msgpack/msgpack/dist.es5+esm/timestamp.mjs
var EXT_TIMESTAMP = -1;
var TIMESTAMP32_MAX_SEC = 4294967296 - 1;
var TIMESTAMP64_MAX_SEC = 17179869184 - 1;
function encodeTimeSpecToTimestamp(_a2) {
  var sec = _a2.sec, nsec = _a2.nsec;
  if (sec >= 0 && nsec >= 0 && sec <= TIMESTAMP64_MAX_SEC) {
    if (nsec === 0 && sec <= TIMESTAMP32_MAX_SEC) {
      var rv = new Uint8Array(4);
      var view = new DataView(rv.buffer);
      view.setUint32(0, sec);
      return rv;
    } else {
      var secHigh = sec / 4294967296;
      var secLow = sec & 4294967295;
      var rv = new Uint8Array(8);
      var view = new DataView(rv.buffer);
      view.setUint32(0, nsec << 2 | secHigh & 3);
      view.setUint32(4, secLow);
      return rv;
    }
  } else {
    var rv = new Uint8Array(12);
    var view = new DataView(rv.buffer);
    view.setUint32(0, nsec);
    setInt64(view, 4, sec);
    return rv;
  }
}
function encodeDateToTimeSpec(date) {
  var msec = date.getTime();
  var sec = Math.floor(msec / 1e3);
  var nsec = (msec - sec * 1e3) * 1e6;
  var nsecInSec = Math.floor(nsec / 1e9);
  return {
    sec: sec + nsecInSec,
    nsec: nsec - nsecInSec * 1e9
  };
}
function encodeTimestampExtension(object) {
  if (object instanceof Date) {
    var timeSpec = encodeDateToTimeSpec(object);
    return encodeTimeSpecToTimestamp(timeSpec);
  } else {
    return null;
  }
}
function decodeTimestampToTimeSpec(data) {
  var view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  switch (data.byteLength) {
    case 4: {
      var sec = view.getUint32(0);
      var nsec = 0;
      return { sec, nsec };
    }
    case 8: {
      var nsec30AndSecHigh2 = view.getUint32(0);
      var secLow32 = view.getUint32(4);
      var sec = (nsec30AndSecHigh2 & 3) * 4294967296 + secLow32;
      var nsec = nsec30AndSecHigh2 >>> 2;
      return { sec, nsec };
    }
    case 12: {
      var sec = getInt64(view, 4);
      var nsec = view.getUint32(0);
      return { sec, nsec };
    }
    default:
      throw new DecodeError("Unrecognized data size for timestamp (expected 4, 8, or 12): ".concat(data.length));
  }
}
function decodeTimestampExtension(data) {
  var timeSpec = decodeTimestampToTimeSpec(data);
  return new Date(timeSpec.sec * 1e3 + timeSpec.nsec / 1e6);
}
var timestampExtension = {
  type: EXT_TIMESTAMP,
  encode: encodeTimestampExtension,
  decode: decodeTimestampExtension
};

// node_modules/@msgpack/msgpack/dist.es5+esm/ExtensionCodec.mjs
var ExtensionCodec = (
  /** @class */
  (function() {
    function ExtensionCodec2() {
      this.builtInEncoders = [];
      this.builtInDecoders = [];
      this.encoders = [];
      this.decoders = [];
      this.register(timestampExtension);
    }
    ExtensionCodec2.prototype.register = function(_a2) {
      var type = _a2.type, encode = _a2.encode, decode2 = _a2.decode;
      if (type >= 0) {
        this.encoders[type] = encode;
        this.decoders[type] = decode2;
      } else {
        var index = 1 + type;
        this.builtInEncoders[index] = encode;
        this.builtInDecoders[index] = decode2;
      }
    };
    ExtensionCodec2.prototype.tryToEncode = function(object, context) {
      for (var i = 0; i < this.builtInEncoders.length; i++) {
        var encodeExt = this.builtInEncoders[i];
        if (encodeExt != null) {
          var data = encodeExt(object, context);
          if (data != null) {
            var type = -1 - i;
            return new ExtData(type, data);
          }
        }
      }
      for (var i = 0; i < this.encoders.length; i++) {
        var encodeExt = this.encoders[i];
        if (encodeExt != null) {
          var data = encodeExt(object, context);
          if (data != null) {
            var type = i;
            return new ExtData(type, data);
          }
        }
      }
      if (object instanceof ExtData) {
        return object;
      }
      return null;
    };
    ExtensionCodec2.prototype.decode = function(data, type, context) {
      var decodeExt = type < 0 ? this.builtInDecoders[-1 - type] : this.decoders[type];
      if (decodeExt) {
        return decodeExt(data, type, context);
      } else {
        return new ExtData(type, data);
      }
    };
    ExtensionCodec2.defaultCodec = new ExtensionCodec2();
    return ExtensionCodec2;
  })()
);

// node_modules/@msgpack/msgpack/dist.es5+esm/utils/typedArrays.mjs
function ensureUint8Array(buffer) {
  if (buffer instanceof Uint8Array) {
    return buffer;
  } else if (ArrayBuffer.isView(buffer)) {
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  } else if (buffer instanceof ArrayBuffer) {
    return new Uint8Array(buffer);
  } else {
    return Uint8Array.from(buffer);
  }
}
function createDataView(buffer) {
  if (buffer instanceof ArrayBuffer) {
    return new DataView(buffer);
  }
  var bufferView = ensureUint8Array(buffer);
  return new DataView(bufferView.buffer, bufferView.byteOffset, bufferView.byteLength);
}

// node_modules/@msgpack/msgpack/dist.es5+esm/utils/prettyByte.mjs
function prettyByte(byte) {
  return "".concat(byte < 0 ? "-" : "", "0x").concat(Math.abs(byte).toString(16).padStart(2, "0"));
}

// node_modules/@msgpack/msgpack/dist.es5+esm/CachedKeyDecoder.mjs
var DEFAULT_MAX_KEY_LENGTH = 16;
var DEFAULT_MAX_LENGTH_PER_KEY = 16;
var CachedKeyDecoder = (
  /** @class */
  (function() {
    function CachedKeyDecoder2(maxKeyLength, maxLengthPerKey) {
      if (maxKeyLength === void 0) {
        maxKeyLength = DEFAULT_MAX_KEY_LENGTH;
      }
      if (maxLengthPerKey === void 0) {
        maxLengthPerKey = DEFAULT_MAX_LENGTH_PER_KEY;
      }
      this.maxKeyLength = maxKeyLength;
      this.maxLengthPerKey = maxLengthPerKey;
      this.hit = 0;
      this.miss = 0;
      this.caches = [];
      for (var i = 0; i < this.maxKeyLength; i++) {
        this.caches.push([]);
      }
    }
    CachedKeyDecoder2.prototype.canBeCached = function(byteLength) {
      return byteLength > 0 && byteLength <= this.maxKeyLength;
    };
    CachedKeyDecoder2.prototype.find = function(bytes, inputOffset, byteLength) {
      var records = this.caches[byteLength - 1];
      FIND_CHUNK: for (var _i = 0, records_1 = records; _i < records_1.length; _i++) {
        var record = records_1[_i];
        var recordBytes = record.bytes;
        for (var j = 0; j < byteLength; j++) {
          if (recordBytes[j] !== bytes[inputOffset + j]) {
            continue FIND_CHUNK;
          }
        }
        return record.str;
      }
      return null;
    };
    CachedKeyDecoder2.prototype.store = function(bytes, value) {
      var records = this.caches[bytes.length - 1];
      var record = { bytes, str: value };
      if (records.length >= this.maxLengthPerKey) {
        records[Math.random() * records.length | 0] = record;
      } else {
        records.push(record);
      }
    };
    CachedKeyDecoder2.prototype.decode = function(bytes, inputOffset, byteLength) {
      var cachedValue = this.find(bytes, inputOffset, byteLength);
      if (cachedValue != null) {
        this.hit++;
        return cachedValue;
      }
      this.miss++;
      var str = utf8DecodeJs(bytes, inputOffset, byteLength);
      var slicedCopyOfBytes = Uint8Array.prototype.slice.call(bytes, inputOffset, inputOffset + byteLength);
      this.store(slicedCopyOfBytes, str);
      return str;
    };
    return CachedKeyDecoder2;
  })()
);

// node_modules/@msgpack/msgpack/dist.es5+esm/Decoder.mjs
var __awaiter = function(thisArg, _arguments, P, generator) {
  function adopt(value) {
    return value instanceof P ? value : new P(function(resolve) {
      resolve(value);
    });
  }
  return new (P || (P = Promise))(function(resolve, reject) {
    function fulfilled(value) {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    }
    function rejected(value) {
      try {
        step(generator["throw"](value));
      } catch (e) {
        reject(e);
      }
    }
    function step(result) {
      result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
    }
    step((generator = generator.apply(thisArg, _arguments || [])).next());
  });
};
var __generator = function(thisArg, body) {
  var _ = { label: 0, sent: function() {
    if (t[0] & 1) throw t[1];
    return t[1];
  }, trys: [], ops: [] }, f, y, t, g;
  return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() {
    return this;
  }), g;
  function verb(n) {
    return function(v) {
      return step([n, v]);
    };
  }
  function step(op) {
    if (f) throw new TypeError("Generator is already executing.");
    while (_) try {
      if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
      if (y = 0, t) op = [op[0] & 2, t.value];
      switch (op[0]) {
        case 0:
        case 1:
          t = op;
          break;
        case 4:
          _.label++;
          return { value: op[1], done: false };
        case 5:
          _.label++;
          y = op[1];
          op = [0];
          continue;
        case 7:
          op = _.ops.pop();
          _.trys.pop();
          continue;
        default:
          if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) {
            _ = 0;
            continue;
          }
          if (op[0] === 3 && (!t || op[1] > t[0] && op[1] < t[3])) {
            _.label = op[1];
            break;
          }
          if (op[0] === 6 && _.label < t[1]) {
            _.label = t[1];
            t = op;
            break;
          }
          if (t && _.label < t[2]) {
            _.label = t[2];
            _.ops.push(op);
            break;
          }
          if (t[2]) _.ops.pop();
          _.trys.pop();
          continue;
      }
      op = body.call(thisArg, _);
    } catch (e) {
      op = [6, e];
      y = 0;
    } finally {
      f = t = 0;
    }
    if (op[0] & 5) throw op[1];
    return { value: op[0] ? op[1] : void 0, done: true };
  }
};
var __asyncValues = function(o) {
  if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
  var m = o[Symbol.asyncIterator], i;
  return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function() {
    return this;
  }, i);
  function verb(n) {
    i[n] = o[n] && function(v) {
      return new Promise(function(resolve, reject) {
        v = o[n](v), settle(resolve, reject, v.done, v.value);
      });
    };
  }
  function settle(resolve, reject, d, v) {
    Promise.resolve(v).then(function(v2) {
      resolve({ value: v2, done: d });
    }, reject);
  }
};
var __await = function(v) {
  return this instanceof __await ? (this.v = v, this) : new __await(v);
};
var __asyncGenerator = function(thisArg, _arguments, generator) {
  if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
  var g = generator.apply(thisArg, _arguments || []), i, q = [];
  return i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function() {
    return this;
  }, i;
  function verb(n) {
    if (g[n]) i[n] = function(v) {
      return new Promise(function(a, b) {
        q.push([n, v, a, b]) > 1 || resume(n, v);
      });
    };
  }
  function resume(n, v) {
    try {
      step(g[n](v));
    } catch (e) {
      settle(q[0][3], e);
    }
  }
  function step(r) {
    r.value instanceof __await ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r);
  }
  function fulfill(value) {
    resume("next", value);
  }
  function reject(value) {
    resume("throw", value);
  }
  function settle(f, v) {
    if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]);
  }
};
var isValidMapKeyType = function(key) {
  var keyType = typeof key;
  return keyType === "string" || keyType === "number";
};
var HEAD_BYTE_REQUIRED = -1;
var EMPTY_VIEW = new DataView(new ArrayBuffer(0));
var EMPTY_BYTES = new Uint8Array(EMPTY_VIEW.buffer);
var DataViewIndexOutOfBoundsError = (function() {
  try {
    EMPTY_VIEW.getInt8(0);
  } catch (e) {
    return e.constructor;
  }
  throw new Error("never reached");
})();
var MORE_DATA = new DataViewIndexOutOfBoundsError("Insufficient data");
var sharedCachedKeyDecoder = new CachedKeyDecoder();
var Decoder = (
  /** @class */
  (function() {
    function Decoder2(extensionCodec, context, maxStrLength, maxBinLength, maxArrayLength, maxMapLength, maxExtLength, keyDecoder) {
      if (extensionCodec === void 0) {
        extensionCodec = ExtensionCodec.defaultCodec;
      }
      if (context === void 0) {
        context = void 0;
      }
      if (maxStrLength === void 0) {
        maxStrLength = UINT32_MAX;
      }
      if (maxBinLength === void 0) {
        maxBinLength = UINT32_MAX;
      }
      if (maxArrayLength === void 0) {
        maxArrayLength = UINT32_MAX;
      }
      if (maxMapLength === void 0) {
        maxMapLength = UINT32_MAX;
      }
      if (maxExtLength === void 0) {
        maxExtLength = UINT32_MAX;
      }
      if (keyDecoder === void 0) {
        keyDecoder = sharedCachedKeyDecoder;
      }
      this.extensionCodec = extensionCodec;
      this.context = context;
      this.maxStrLength = maxStrLength;
      this.maxBinLength = maxBinLength;
      this.maxArrayLength = maxArrayLength;
      this.maxMapLength = maxMapLength;
      this.maxExtLength = maxExtLength;
      this.keyDecoder = keyDecoder;
      this.totalPos = 0;
      this.pos = 0;
      this.view = EMPTY_VIEW;
      this.bytes = EMPTY_BYTES;
      this.headByte = HEAD_BYTE_REQUIRED;
      this.stack = [];
    }
    Decoder2.prototype.reinitializeState = function() {
      this.totalPos = 0;
      this.headByte = HEAD_BYTE_REQUIRED;
      this.stack.length = 0;
    };
    Decoder2.prototype.setBuffer = function(buffer) {
      this.bytes = ensureUint8Array(buffer);
      this.view = createDataView(this.bytes);
      this.pos = 0;
    };
    Decoder2.prototype.appendBuffer = function(buffer) {
      if (this.headByte === HEAD_BYTE_REQUIRED && !this.hasRemaining(1)) {
        this.setBuffer(buffer);
      } else {
        var remainingData = this.bytes.subarray(this.pos);
        var newData = ensureUint8Array(buffer);
        var newBuffer = new Uint8Array(remainingData.length + newData.length);
        newBuffer.set(remainingData);
        newBuffer.set(newData, remainingData.length);
        this.setBuffer(newBuffer);
      }
    };
    Decoder2.prototype.hasRemaining = function(size) {
      return this.view.byteLength - this.pos >= size;
    };
    Decoder2.prototype.createExtraByteError = function(posToShow) {
      var _a2 = this, view = _a2.view, pos = _a2.pos;
      return new RangeError("Extra ".concat(view.byteLength - pos, " of ").concat(view.byteLength, " byte(s) found at buffer[").concat(posToShow, "]"));
    };
    Decoder2.prototype.decode = function(buffer) {
      this.reinitializeState();
      this.setBuffer(buffer);
      var object = this.doDecodeSync();
      if (this.hasRemaining(1)) {
        throw this.createExtraByteError(this.pos);
      }
      return object;
    };
    Decoder2.prototype.decodeMulti = function(buffer) {
      return __generator(this, function(_a2) {
        switch (_a2.label) {
          case 0:
            this.reinitializeState();
            this.setBuffer(buffer);
            _a2.label = 1;
          case 1:
            if (!this.hasRemaining(1)) return [3, 3];
            return [4, this.doDecodeSync()];
          case 2:
            _a2.sent();
            return [3, 1];
          case 3:
            return [
              2
              /*return*/
            ];
        }
      });
    };
    Decoder2.prototype.decodeAsync = function(stream) {
      var stream_1, stream_1_1;
      var e_1, _a2;
      return __awaiter(this, void 0, void 0, function() {
        var decoded, object, buffer, e_1_1, _b2, headByte, pos, totalPos;
        return __generator(this, function(_c2) {
          switch (_c2.label) {
            case 0:
              decoded = false;
              _c2.label = 1;
            case 1:
              _c2.trys.push([1, 6, 7, 12]);
              stream_1 = __asyncValues(stream);
              _c2.label = 2;
            case 2:
              return [4, stream_1.next()];
            case 3:
              if (!(stream_1_1 = _c2.sent(), !stream_1_1.done)) return [3, 5];
              buffer = stream_1_1.value;
              if (decoded) {
                throw this.createExtraByteError(this.totalPos);
              }
              this.appendBuffer(buffer);
              try {
                object = this.doDecodeSync();
                decoded = true;
              } catch (e) {
                if (!(e instanceof DataViewIndexOutOfBoundsError)) {
                  throw e;
                }
              }
              this.totalPos += this.pos;
              _c2.label = 4;
            case 4:
              return [3, 2];
            case 5:
              return [3, 12];
            case 6:
              e_1_1 = _c2.sent();
              e_1 = { error: e_1_1 };
              return [3, 12];
            case 7:
              _c2.trys.push([7, , 10, 11]);
              if (!(stream_1_1 && !stream_1_1.done && (_a2 = stream_1.return))) return [3, 9];
              return [4, _a2.call(stream_1)];
            case 8:
              _c2.sent();
              _c2.label = 9;
            case 9:
              return [3, 11];
            case 10:
              if (e_1) throw e_1.error;
              return [
                7
                /*endfinally*/
              ];
            case 11:
              return [
                7
                /*endfinally*/
              ];
            case 12:
              if (decoded) {
                if (this.hasRemaining(1)) {
                  throw this.createExtraByteError(this.totalPos);
                }
                return [2, object];
              }
              _b2 = this, headByte = _b2.headByte, pos = _b2.pos, totalPos = _b2.totalPos;
              throw new RangeError("Insufficient data in parsing ".concat(prettyByte(headByte), " at ").concat(totalPos, " (").concat(pos, " in the current buffer)"));
          }
        });
      });
    };
    Decoder2.prototype.decodeArrayStream = function(stream) {
      return this.decodeMultiAsync(stream, true);
    };
    Decoder2.prototype.decodeStream = function(stream) {
      return this.decodeMultiAsync(stream, false);
    };
    Decoder2.prototype.decodeMultiAsync = function(stream, isArray) {
      return __asyncGenerator(this, arguments, function decodeMultiAsync_1() {
        var isArrayHeaderRequired, arrayItemsLeft, stream_2, stream_2_1, buffer, e_2, e_3_1;
        var e_3, _a2;
        return __generator(this, function(_b2) {
          switch (_b2.label) {
            case 0:
              isArrayHeaderRequired = isArray;
              arrayItemsLeft = -1;
              _b2.label = 1;
            case 1:
              _b2.trys.push([1, 13, 14, 19]);
              stream_2 = __asyncValues(stream);
              _b2.label = 2;
            case 2:
              return [4, __await(stream_2.next())];
            case 3:
              if (!(stream_2_1 = _b2.sent(), !stream_2_1.done)) return [3, 12];
              buffer = stream_2_1.value;
              if (isArray && arrayItemsLeft === 0) {
                throw this.createExtraByteError(this.totalPos);
              }
              this.appendBuffer(buffer);
              if (isArrayHeaderRequired) {
                arrayItemsLeft = this.readArraySize();
                isArrayHeaderRequired = false;
                this.complete();
              }
              _b2.label = 4;
            case 4:
              _b2.trys.push([4, 9, , 10]);
              _b2.label = 5;
            case 5:
              if (false) return [3, 8];
              return [4, __await(this.doDecodeSync())];
            case 6:
              return [4, _b2.sent()];
            case 7:
              _b2.sent();
              if (--arrayItemsLeft === 0) {
                return [3, 8];
              }
              return [3, 5];
            case 8:
              return [3, 10];
            case 9:
              e_2 = _b2.sent();
              if (!(e_2 instanceof DataViewIndexOutOfBoundsError)) {
                throw e_2;
              }
              return [3, 10];
            case 10:
              this.totalPos += this.pos;
              _b2.label = 11;
            case 11:
              return [3, 2];
            case 12:
              return [3, 19];
            case 13:
              e_3_1 = _b2.sent();
              e_3 = { error: e_3_1 };
              return [3, 19];
            case 14:
              _b2.trys.push([14, , 17, 18]);
              if (!(stream_2_1 && !stream_2_1.done && (_a2 = stream_2.return))) return [3, 16];
              return [4, __await(_a2.call(stream_2))];
            case 15:
              _b2.sent();
              _b2.label = 16;
            case 16:
              return [3, 18];
            case 17:
              if (e_3) throw e_3.error;
              return [
                7
                /*endfinally*/
              ];
            case 18:
              return [
                7
                /*endfinally*/
              ];
            case 19:
              return [
                2
                /*return*/
              ];
          }
        });
      });
    };
    Decoder2.prototype.doDecodeSync = function() {
      DECODE: while (true) {
        var headByte = this.readHeadByte();
        var object = void 0;
        if (headByte >= 224) {
          object = headByte - 256;
        } else if (headByte < 192) {
          if (headByte < 128) {
            object = headByte;
          } else if (headByte < 144) {
            var size = headByte - 128;
            if (size !== 0) {
              this.pushMapState(size);
              this.complete();
              continue DECODE;
            } else {
              object = {};
            }
          } else if (headByte < 160) {
            var size = headByte - 144;
            if (size !== 0) {
              this.pushArrayState(size);
              this.complete();
              continue DECODE;
            } else {
              object = [];
            }
          } else {
            var byteLength = headByte - 160;
            object = this.decodeUtf8String(byteLength, 0);
          }
        } else if (headByte === 192) {
          object = null;
        } else if (headByte === 194) {
          object = false;
        } else if (headByte === 195) {
          object = true;
        } else if (headByte === 202) {
          object = this.readF32();
        } else if (headByte === 203) {
          object = this.readF64();
        } else if (headByte === 204) {
          object = this.readU8();
        } else if (headByte === 205) {
          object = this.readU16();
        } else if (headByte === 206) {
          object = this.readU32();
        } else if (headByte === 207) {
          object = this.readU64();
        } else if (headByte === 208) {
          object = this.readI8();
        } else if (headByte === 209) {
          object = this.readI16();
        } else if (headByte === 210) {
          object = this.readI32();
        } else if (headByte === 211) {
          object = this.readI64();
        } else if (headByte === 217) {
          var byteLength = this.lookU8();
          object = this.decodeUtf8String(byteLength, 1);
        } else if (headByte === 218) {
          var byteLength = this.lookU16();
          object = this.decodeUtf8String(byteLength, 2);
        } else if (headByte === 219) {
          var byteLength = this.lookU32();
          object = this.decodeUtf8String(byteLength, 4);
        } else if (headByte === 220) {
          var size = this.readU16();
          if (size !== 0) {
            this.pushArrayState(size);
            this.complete();
            continue DECODE;
          } else {
            object = [];
          }
        } else if (headByte === 221) {
          var size = this.readU32();
          if (size !== 0) {
            this.pushArrayState(size);
            this.complete();
            continue DECODE;
          } else {
            object = [];
          }
        } else if (headByte === 222) {
          var size = this.readU16();
          if (size !== 0) {
            this.pushMapState(size);
            this.complete();
            continue DECODE;
          } else {
            object = {};
          }
        } else if (headByte === 223) {
          var size = this.readU32();
          if (size !== 0) {
            this.pushMapState(size);
            this.complete();
            continue DECODE;
          } else {
            object = {};
          }
        } else if (headByte === 196) {
          var size = this.lookU8();
          object = this.decodeBinary(size, 1);
        } else if (headByte === 197) {
          var size = this.lookU16();
          object = this.decodeBinary(size, 2);
        } else if (headByte === 198) {
          var size = this.lookU32();
          object = this.decodeBinary(size, 4);
        } else if (headByte === 212) {
          object = this.decodeExtension(1, 0);
        } else if (headByte === 213) {
          object = this.decodeExtension(2, 0);
        } else if (headByte === 214) {
          object = this.decodeExtension(4, 0);
        } else if (headByte === 215) {
          object = this.decodeExtension(8, 0);
        } else if (headByte === 216) {
          object = this.decodeExtension(16, 0);
        } else if (headByte === 199) {
          var size = this.lookU8();
          object = this.decodeExtension(size, 1);
        } else if (headByte === 200) {
          var size = this.lookU16();
          object = this.decodeExtension(size, 2);
        } else if (headByte === 201) {
          var size = this.lookU32();
          object = this.decodeExtension(size, 4);
        } else {
          throw new DecodeError("Unrecognized type byte: ".concat(prettyByte(headByte)));
        }
        this.complete();
        var stack = this.stack;
        while (stack.length > 0) {
          var state = stack[stack.length - 1];
          if (state.type === 0) {
            state.array[state.position] = object;
            state.position++;
            if (state.position === state.size) {
              stack.pop();
              object = state.array;
            } else {
              continue DECODE;
            }
          } else if (state.type === 1) {
            if (!isValidMapKeyType(object)) {
              throw new DecodeError("The type of key must be string or number but " + typeof object);
            }
            if (object === "__proto__") {
              throw new DecodeError("The key __proto__ is not allowed");
            }
            state.key = object;
            state.type = 2;
            continue DECODE;
          } else {
            state.map[state.key] = object;
            state.readCount++;
            if (state.readCount === state.size) {
              stack.pop();
              object = state.map;
            } else {
              state.key = null;
              state.type = 1;
              continue DECODE;
            }
          }
        }
        return object;
      }
    };
    Decoder2.prototype.readHeadByte = function() {
      if (this.headByte === HEAD_BYTE_REQUIRED) {
        this.headByte = this.readU8();
      }
      return this.headByte;
    };
    Decoder2.prototype.complete = function() {
      this.headByte = HEAD_BYTE_REQUIRED;
    };
    Decoder2.prototype.readArraySize = function() {
      var headByte = this.readHeadByte();
      switch (headByte) {
        case 220:
          return this.readU16();
        case 221:
          return this.readU32();
        default: {
          if (headByte < 160) {
            return headByte - 144;
          } else {
            throw new DecodeError("Unrecognized array type byte: ".concat(prettyByte(headByte)));
          }
        }
      }
    };
    Decoder2.prototype.pushMapState = function(size) {
      if (size > this.maxMapLength) {
        throw new DecodeError("Max length exceeded: map length (".concat(size, ") > maxMapLengthLength (").concat(this.maxMapLength, ")"));
      }
      this.stack.push({
        type: 1,
        size,
        key: null,
        readCount: 0,
        map: {}
      });
    };
    Decoder2.prototype.pushArrayState = function(size) {
      if (size > this.maxArrayLength) {
        throw new DecodeError("Max length exceeded: array length (".concat(size, ") > maxArrayLength (").concat(this.maxArrayLength, ")"));
      }
      this.stack.push({
        type: 0,
        size,
        array: new Array(size),
        position: 0
      });
    };
    Decoder2.prototype.decodeUtf8String = function(byteLength, headerOffset) {
      var _a2;
      if (byteLength > this.maxStrLength) {
        throw new DecodeError("Max length exceeded: UTF-8 byte length (".concat(byteLength, ") > maxStrLength (").concat(this.maxStrLength, ")"));
      }
      if (this.bytes.byteLength < this.pos + headerOffset + byteLength) {
        throw MORE_DATA;
      }
      var offset = this.pos + headerOffset;
      var object;
      if (this.stateIsMapKey() && ((_a2 = this.keyDecoder) === null || _a2 === void 0 ? void 0 : _a2.canBeCached(byteLength))) {
        object = this.keyDecoder.decode(this.bytes, offset, byteLength);
      } else if (byteLength > TEXT_DECODER_THRESHOLD) {
        object = utf8DecodeTD(this.bytes, offset, byteLength);
      } else {
        object = utf8DecodeJs(this.bytes, offset, byteLength);
      }
      this.pos += headerOffset + byteLength;
      return object;
    };
    Decoder2.prototype.stateIsMapKey = function() {
      if (this.stack.length > 0) {
        var state = this.stack[this.stack.length - 1];
        return state.type === 1;
      }
      return false;
    };
    Decoder2.prototype.decodeBinary = function(byteLength, headOffset) {
      if (byteLength > this.maxBinLength) {
        throw new DecodeError("Max length exceeded: bin length (".concat(byteLength, ") > maxBinLength (").concat(this.maxBinLength, ")"));
      }
      if (!this.hasRemaining(byteLength + headOffset)) {
        throw MORE_DATA;
      }
      var offset = this.pos + headOffset;
      var object = this.bytes.subarray(offset, offset + byteLength);
      this.pos += headOffset + byteLength;
      return object;
    };
    Decoder2.prototype.decodeExtension = function(size, headOffset) {
      if (size > this.maxExtLength) {
        throw new DecodeError("Max length exceeded: ext length (".concat(size, ") > maxExtLength (").concat(this.maxExtLength, ")"));
      }
      var extType = this.view.getInt8(this.pos + headOffset);
      var data = this.decodeBinary(
        size,
        headOffset + 1
        /* extType */
      );
      return this.extensionCodec.decode(data, extType, this.context);
    };
    Decoder2.prototype.lookU8 = function() {
      return this.view.getUint8(this.pos);
    };
    Decoder2.prototype.lookU16 = function() {
      return this.view.getUint16(this.pos);
    };
    Decoder2.prototype.lookU32 = function() {
      return this.view.getUint32(this.pos);
    };
    Decoder2.prototype.readU8 = function() {
      var value = this.view.getUint8(this.pos);
      this.pos++;
      return value;
    };
    Decoder2.prototype.readI8 = function() {
      var value = this.view.getInt8(this.pos);
      this.pos++;
      return value;
    };
    Decoder2.prototype.readU16 = function() {
      var value = this.view.getUint16(this.pos);
      this.pos += 2;
      return value;
    };
    Decoder2.prototype.readI16 = function() {
      var value = this.view.getInt16(this.pos);
      this.pos += 2;
      return value;
    };
    Decoder2.prototype.readU32 = function() {
      var value = this.view.getUint32(this.pos);
      this.pos += 4;
      return value;
    };
    Decoder2.prototype.readI32 = function() {
      var value = this.view.getInt32(this.pos);
      this.pos += 4;
      return value;
    };
    Decoder2.prototype.readU64 = function() {
      var value = getUint64(this.view, this.pos);
      this.pos += 8;
      return value;
    };
    Decoder2.prototype.readI64 = function() {
      var value = getInt64(this.view, this.pos);
      this.pos += 8;
      return value;
    };
    Decoder2.prototype.readF32 = function() {
      var value = this.view.getFloat32(this.pos);
      this.pos += 4;
      return value;
    };
    Decoder2.prototype.readF64 = function() {
      var value = this.view.getFloat64(this.pos);
      this.pos += 8;
      return value;
    };
    return Decoder2;
  })()
);

// node_modules/@msgpack/msgpack/dist.es5+esm/decode.mjs
var defaultDecodeOptions = {};
function decode(buffer, options) {
  if (options === void 0) {
    options = defaultDecodeOptions;
  }
  var decoder = new Decoder(options.extensionCodec, options.context, options.maxStrLength, options.maxBinLength, options.maxArrayLength, options.maxMapLength, options.maxExtLength);
  return decoder.decode(buffer);
}

// src/utils/index.ts
function decodeRecord(record) {
  const recordEntry = record.entry;
  if (!recordEntry) {
    throw new Error("Record has no entry");
  }
  if ("Present" in recordEntry) {
    const entry = recordEntry.Present;
    if (entry.entry_type === "App") {
      return decode(entry.entry);
    }
    throw new Error(`Unexpected entry type: ${entry.entry_type}`);
  }
  if ("Hidden" in recordEntry) {
    throw new Error("Entry is hidden");
  }
  if ("NotApplicable" in recordEntry) {
    throw new Error("Entry not applicable");
  }
  if ("NotStored" in recordEntry) {
    throw new Error("Entry not stored");
  }
  throw new Error("Entry not present in record");
}
function decodeRecords(records) {
  return records.map((record) => decodeRecord(record));
}
function epistemicCode(e, n, m, h) {
  return `E${e}N${n}M${m}H${h}`;
}
function calculateEpistemicStrength(e, n, m, h) {
  const eVal = e / 4;
  const nVal = n / 3;
  const mVal = m / 3;
  const hVal = h / 4;
  return 0.4 * eVal + 0.25 * nVal + 0.2 * mVal + 0.15 * hVal;
}
function formatTimestamp(timestamp) {
  const date = new Date(timestamp / 1e3);
  return date.toISOString();
}
function parseToTimestamp(isoDate) {
  return new Date(isoDate).getTime() * 1e3;
}
var SOURCE_QUALITY_WEIGHTS = {
  AcademicPaper: 1,
  OfficialDocument: 0.95,
  Dataset: 0.9,
  Book: 0.85,
  KnowledgeBase: 0.8,
  NewsArticle: 0.6,
  Video: 0.5,
  Audio: 0.5,
  BlogPost: 0.4,
  WebPage: 0.35,
  LucidThought: 0.5,
  Conversation: 0.3,
  PersonalExperience: 0.3,
  SocialMedia: 0.2,
  Other: 0.25
};
function getSourceQualityWeight(sourceType) {
  return SOURCE_QUALITY_WEIGHTS[sourceType] ?? 0.25;
}

// src/zomes/lucid.ts
var LucidZomeClient = class {
  constructor(client, roleName = "lucid", zomeName = "lucid") {
    this.client = client;
    this.roleName = roleName;
    this.zomeName = zomeName;
  }
  async callZome(fnName, payload) {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null
    });
    return result;
  }
  // ============================================================================
  // THOUGHT OPERATIONS
  // ============================================================================
  /** Create a new thought */
  async createThought(input) {
    const record = await this.callZome("create_thought", input);
    return decodeRecord(record);
  }
  /** Get a thought by ID */
  async getThought(thoughtId) {
    const record = await this.callZome("get_thought", thoughtId);
    return record ? decodeRecord(record) : null;
  }
  /** Get a thought by action hash */
  async getThoughtByHash(actionHash) {
    const record = await this.callZome("get_thought_by_hash", actionHash);
    return record ? decodeRecord(record) : null;
  }
  /** Update an existing thought */
  async updateThought(input) {
    const record = await this.callZome("update_thought", input);
    return decodeRecord(record);
  }
  /** Delete a thought */
  async deleteThought(thoughtId) {
    return this.callZome("delete_thought", thoughtId);
  }
  /** Get all thoughts for the current user */
  async getMyThoughts() {
    const records = await this.callZome("get_my_thoughts", null);
    return decodeRecords(records);
  }
  /** Get thoughts by tag */
  async getThoughtsByTag(tag) {
    const records = await this.callZome("get_thoughts_by_tag", tag);
    return decodeRecords(records);
  }
  /** Get thoughts by domain */
  async getThoughtsByDomain(domain) {
    const records = await this.callZome("get_thoughts_by_domain", domain);
    return decodeRecords(records);
  }
  /** Get thoughts by type */
  async getThoughtsByType(thoughtType) {
    const records = await this.callZome("get_thoughts_by_type", thoughtType);
    return decodeRecords(records);
  }
  /** Get child thoughts of a parent */
  async getChildThoughts(parentThoughtId) {
    const records = await this.callZome("get_child_thoughts", parentThoughtId);
    return decodeRecords(records);
  }
  /** Search thoughts with filters */
  async searchThoughts(input) {
    const records = await this.callZome("search_thoughts", input);
    return decodeRecords(records);
  }
  // ============================================================================
  // TAG OPERATIONS
  // ============================================================================
  /** Create a new tag */
  async createTag(input) {
    const record = await this.callZome("create_tag", input);
    return decodeRecord(record);
  }
  /** Get all tags for the current user */
  async getMyTags() {
    const records = await this.callZome("get_my_tags", null);
    return decodeRecords(records);
  }
  // ============================================================================
  // DOMAIN OPERATIONS
  // ============================================================================
  /** Create a new domain */
  async createDomain(input) {
    const record = await this.callZome("create_domain", input);
    return decodeRecord(record);
  }
  /** Get all domains for the current user */
  async getMyDomains() {
    const records = await this.callZome("get_my_domains", null);
    return decodeRecords(records);
  }
  // ============================================================================
  // RELATIONSHIP OPERATIONS
  // ============================================================================
  /** Create a relationship between thoughts */
  async createRelationship(input) {
    const record = await this.callZome("create_relationship", input);
    return decodeRecord(record);
  }
  /** Get relationships from a thought */
  async getThoughtRelationships(thoughtId) {
    const records = await this.callZome("get_thought_relationships", thoughtId);
    return decodeRecords(records);
  }
  /** Get thoughts related to a given thought */
  async getRelatedThoughts(thoughtId) {
    const records = await this.callZome("get_related_thoughts", thoughtId);
    return decodeRecords(records);
  }
  // ============================================================================
  // STATISTICS
  // ============================================================================
  /** Get statistics about the knowledge graph */
  async getStats() {
    return this.callZome("get_stats", null);
  }
};

// src/zomes/sources.ts
var SourcesZomeClient = class {
  constructor(client, roleName = "lucid", zomeName = "sources") {
    this.client = client;
    this.roleName = roleName;
    this.zomeName = zomeName;
  }
  async callZome(fnName, payload) {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null
    });
    return result;
  }
  // ============================================================================
  // SOURCE OPERATIONS
  // ============================================================================
  /** Create a new source */
  async createSource(input) {
    const record = await this.callZome("create_source", input);
    return decodeRecord(record);
  }
  /** Get a source by ID */
  async getSource(sourceId) {
    const record = await this.callZome("get_source", sourceId);
    return record ? decodeRecord(record) : null;
  }
  /** Get a source by URL */
  async getSourceByUrl(url) {
    const record = await this.callZome("get_source_by_url", url);
    return record ? decodeRecord(record) : null;
  }
  /** Get all sources for the current user */
  async getMySources() {
    const records = await this.callZome("get_my_sources", null);
    return decodeRecords(records);
  }
  /** Get sources by type */
  async getSourcesByType(sourceType) {
    const records = await this.callZome("get_sources_by_type", sourceType);
    return decodeRecords(records);
  }
  // ============================================================================
  // CITATION OPERATIONS
  // ============================================================================
  /** Create a citation linking a thought to a source */
  async createCitation(input) {
    const record = await this.callZome("create_citation", input);
    return decodeRecord(record);
  }
  /** Get citations for a thought */
  async getThoughtCitations(thoughtId) {
    const records = await this.callZome("get_thought_citations", thoughtId);
    return decodeRecords(records);
  }
  /** Get citations for a source */
  async getSourceCitations(sourceId) {
    const records = await this.callZome("get_source_citations", sourceId);
    return decodeRecords(records);
  }
  /** Get all sources for a thought (via citations) */
  async getThoughtSources(thoughtId) {
    const records = await this.callZome("get_thought_sources", thoughtId);
    return decodeRecords(records);
  }
};

// src/zomes/temporal.ts
var TemporalZomeClient = class {
  constructor(client, roleName = "lucid", zomeName = "temporal") {
    this.client = client;
    this.roleName = roleName;
    this.zomeName = zomeName;
  }
  async callZome(fnName, payload) {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null
    });
    return result;
  }
  // ============================================================================
  // VERSION OPERATIONS
  // ============================================================================
  /** Record a new version of a thought */
  async recordVersion(input) {
    const record = await this.callZome("record_version", input);
    return decodeRecord(record);
  }
  /** Get all versions of a thought (history) */
  async getThoughtHistory(thoughtId) {
    const records = await this.callZome("get_thought_history", thoughtId);
    return decodeRecords(records);
  }
  /** Get belief at a specific timestamp */
  async getBeliefAtTime(thoughtId, timestamp) {
    const input = { thought_id: thoughtId, timestamp };
    const record = await this.callZome("get_belief_at_time", input);
    return record ? decodeRecord(record) : null;
  }
  // ============================================================================
  // SNAPSHOT OPERATIONS
  // ============================================================================
  /** Create a snapshot of the current knowledge graph */
  async createSnapshot(input) {
    const record = await this.callZome("create_snapshot", input);
    return decodeRecord(record);
  }
  /** Get all snapshots */
  async getMySnapshots() {
    const records = await this.callZome("get_my_snapshots", null);
    return decodeRecords(records);
  }
};

// src/zomes/privacy.ts
var PrivacyZomeClient = class {
  constructor(client, roleName = "lucid", zomeName = "privacy") {
    this.client = client;
    this.roleName = roleName;
    this.zomeName = zomeName;
  }
  async callZome(fnName, payload) {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null
    });
    return result;
  }
  // ============================================================================
  // POLICY OPERATIONS
  // ============================================================================
  /** Set sharing policy for a thought */
  async setSharingPolicy(input) {
    const record = await this.callZome("set_sharing_policy", input);
    return decodeRecord(record);
  }
  /** Get sharing policy for a thought */
  async getSharingPolicy(thoughtId) {
    const record = await this.callZome("get_sharing_policy", thoughtId);
    return record ? decodeRecord(record) : null;
  }
  // ============================================================================
  // GRANT OPERATIONS
  // ============================================================================
  /** Grant access to a specific agent */
  async grantAccess(input) {
    const record = await this.callZome("grant_access", input);
    return decodeRecord(record);
  }
  /** Get grants for a thought */
  async getThoughtGrants(thoughtId) {
    const records = await this.callZome("get_thought_grants", thoughtId);
    return decodeRecords(records);
  }
  /** Check if an agent has access to a thought */
  async checkAccess(thoughtId, agent) {
    const input = { thought_id: thoughtId, agent };
    return this.callZome("check_access", input);
  }
  /** Log an access event */
  async logAccess(thoughtId, accessType) {
    return this.callZome("log_access", {
      thought_id: thoughtId,
      access_type: accessType
    });
  }
};

// src/zomes/reasoning.ts
var ReasoningZomeClient = class {
  constructor(client, roleName = "lucid", zomeName = "reasoning") {
    this.client = client;
    this.roleName = roleName;
    this.zomeName = zomeName;
  }
  async callZome(fnName, payload) {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null
    });
    return result;
  }
  // ============================================================================
  // CONTRADICTION OPERATIONS
  // ============================================================================
  /** Record a detected contradiction */
  async recordContradiction(input) {
    const record = await this.callZome("record_contradiction", input);
    return decodeRecord(record);
  }
  /** Get contradictions for a thought */
  async getThoughtContradictions(thoughtId) {
    const records = await this.callZome("get_thought_contradictions", thoughtId);
    return decodeRecords(records);
  }
  // ============================================================================
  // COHERENCE OPERATIONS
  // ============================================================================
  /** Create a coherence report */
  async createCoherenceReport(input) {
    const record = await this.callZome("create_coherence_report", input);
    return decodeRecord(record);
  }
  // ============================================================================
  // INFERENCE OPERATIONS
  // ============================================================================
  /** Record an inference */
  async recordInference(input) {
    const record = await this.callZome("record_inference", input);
    return decodeRecord(record);
  }
  /** Get all inferences */
  async getMyInferences() {
    const records = await this.callZome("get_my_inferences", null);
    return decodeRecords(records);
  }
};

// src/zomes/bridge.ts
var BridgeZomeClient = class {
  constructor(client, roleName = "lucid", zomeName = "bridge") {
    this.client = client;
    this.roleName = roleName;
    this.zomeName = zomeName;
  }
  async callZome(fnName, payload) {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null
    });
    return result;
  }
  // ============================================================================
  // FEDERATION OPERATIONS
  // ============================================================================
  /** Record federation of a thought to external hApp */
  async recordFederation(thoughtId, targetHapp, externalId) {
    const record = await this.callZome("record_federation", {
      thought_id: thoughtId,
      target_happ: targetHapp,
      external_id: externalId
    });
    return decodeRecord(record);
  }
  /** Get federation records for a thought */
  async getThoughtFederations(thoughtId) {
    const records = await this.callZome("get_thought_federations", thoughtId);
    return decodeRecords(records);
  }
  // ============================================================================
  // REPUTATION OPERATIONS
  // ============================================================================
  /** Cache external reputation score */
  async cacheReputation(agent, kVector, trustScore, sourceHapp) {
    const record = await this.callZome("cache_reputation", {
      agent,
      k_vector: kVector,
      trust_score: trustScore,
      source_happ: sourceHapp
    });
    return decodeRecord(record);
  }
  /** Get cached reputation for an agent */
  async getCachedReputation(agent) {
    const record = await this.callZome("get_cached_reputation", agent);
    return record ? decodeRecord(record) : null;
  }
  // ============================================================================
  // SYMTHAEA INTEGRATION
  // ============================================================================
  /** Check coherence with Symthaea consciousness engine */
  async checkCoherenceWithSymthaea(thoughtIds) {
    return this.callZome("check_coherence_with_symthaea", thoughtIds);
  }
};

// src/zomes/collective.ts
var CollectiveZomeClient = class {
  constructor(client, roleName = "lucid", zomeName = "collective") {
    this.client = client;
    this.roleName = roleName;
    this.zomeName = zomeName;
  }
  async callZome(fnName, payload) {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null
    });
    return result;
  }
  // ============================================================================
  // BELIEF SHARING
  // ============================================================================
  /** Share a belief to the collective */
  async shareBelief(input) {
    const record = await this.callZome("share_belief", input);
    return decodeRecord(record);
  }
  /** Get all shared beliefs */
  async getAllBeliefShares() {
    const records = await this.callZome("get_all_belief_shares", null);
    return decodeRecords(records);
  }
  /** Get beliefs by tag */
  async getBeliefsByTag(tag) {
    const records = await this.callZome("get_beliefs_by_tag", tag);
    return decodeRecords(records);
  }
  // ============================================================================
  // VOTING
  // ============================================================================
  /** Cast a vote on a shared belief */
  async castVote(input) {
    const record = await this.callZome("cast_vote", input);
    return decodeRecord(record);
  }
  /** Get votes for a belief share */
  async getBeliefVotes(beliefHash) {
    const records = await this.callZome("get_belief_votes", beliefHash);
    return decodeRecords(records);
  }
  // ============================================================================
  // CONSENSUS
  // ============================================================================
  /** Get consensus for a belief */
  async getBeliefConsensus(beliefHash) {
    const record = await this.callZome("get_belief_consensus", beliefHash);
    return record ? decodeRecord(record) : null;
  }
  /** Calculate weighted consensus using relationships */
  async calculateWeightedConsensus(beliefHash) {
    return this.callZome("calculate_weighted_consensus", beliefHash);
  }
  // ============================================================================
  // PATTERN DETECTION
  // ============================================================================
  /** Detect emergent patterns across beliefs */
  async detectPatterns(input) {
    return this.callZome("detect_patterns", input ?? {});
  }
  /** Record an emergent pattern */
  async recordPattern(input) {
    const record = await this.callZome("record_pattern", input);
    return decodeRecord(record);
  }
  /** Get all recorded patterns */
  async getRecordedPatterns() {
    const records = await this.callZome("get_recorded_patterns", null);
    return decodeRecords(records);
  }
  // ============================================================================
  // REPUTATION
  // ============================================================================
  /** Update agent's epistemic reputation */
  async updateReputation(input) {
    const record = await this.callZome("update_reputation", input);
    return decodeRecord(record);
  }
  // ============================================================================
  // RELATIONSHIPS
  // ============================================================================
  /** Update or create a relationship with another agent */
  async updateRelationship(input) {
    const record = await this.callZome("update_relationship", input);
    return decodeRecord(record);
  }
  /** Get relationship with a specific agent */
  async getRelationship(otherAgent) {
    const result = await this.callZome("get_relationship", otherAgent);
    return result;
  }
  /** Get all my relationships */
  async getMyRelationships() {
    return this.callZome("get_my_relationships", null);
  }
  // ============================================================================
  // STATISTICS
  // ============================================================================
  /** Get collective statistics */
  async getCollectiveStats() {
    return this.callZome("get_collective_stats", null);
  }
};

// src/types/index.ts
var EmpiricalLevel = /* @__PURE__ */ ((EmpiricalLevel2) => {
  EmpiricalLevel2["E0"] = "E0";
  EmpiricalLevel2["E1"] = "E1";
  EmpiricalLevel2["E2"] = "E2";
  EmpiricalLevel2["E3"] = "E3";
  EmpiricalLevel2["E4"] = "E4";
  return EmpiricalLevel2;
})(EmpiricalLevel || {});
var NormativeLevel = /* @__PURE__ */ ((NormativeLevel2) => {
  NormativeLevel2["N0"] = "N0";
  NormativeLevel2["N1"] = "N1";
  NormativeLevel2["N2"] = "N2";
  NormativeLevel2["N3"] = "N3";
  return NormativeLevel2;
})(NormativeLevel || {});
var MaterialityLevel = /* @__PURE__ */ ((MaterialityLevel2) => {
  MaterialityLevel2["M0"] = "M0";
  MaterialityLevel2["M1"] = "M1";
  MaterialityLevel2["M2"] = "M2";
  MaterialityLevel2["M3"] = "M3";
  return MaterialityLevel2;
})(MaterialityLevel || {});
var HarmonicLevel = /* @__PURE__ */ ((HarmonicLevel2) => {
  HarmonicLevel2["H0"] = "H0";
  HarmonicLevel2["H1"] = "H1";
  HarmonicLevel2["H2"] = "H2";
  HarmonicLevel2["H3"] = "H3";
  HarmonicLevel2["H4"] = "H4";
  return HarmonicLevel2;
})(HarmonicLevel || {});
var ThoughtType = /* @__PURE__ */ ((ThoughtType2) => {
  ThoughtType2["Claim"] = "Claim";
  ThoughtType2["Note"] = "Note";
  ThoughtType2["Question"] = "Question";
  ThoughtType2["Insight"] = "Insight";
  ThoughtType2["Definition"] = "Definition";
  ThoughtType2["Prediction"] = "Prediction";
  ThoughtType2["Hypothesis"] = "Hypothesis";
  ThoughtType2["Reflection"] = "Reflection";
  ThoughtType2["Quote"] = "Quote";
  ThoughtType2["Task"] = "Task";
  return ThoughtType2;
})(ThoughtType || {});
var RelationshipType = /* @__PURE__ */ ((RelationshipType2) => {
  RelationshipType2["Supports"] = "Supports";
  RelationshipType2["Contradicts"] = "Contradicts";
  RelationshipType2["Implies"] = "Implies";
  RelationshipType2["ImpliedBy"] = "ImpliedBy";
  RelationshipType2["Refines"] = "Refines";
  RelationshipType2["ExampleOf"] = "ExampleOf";
  RelationshipType2["RelatedTo"] = "RelatedTo";
  RelationshipType2["DependsOn"] = "DependsOn";
  RelationshipType2["Supersedes"] = "Supersedes";
  RelationshipType2["RespondsTo"] = "RespondsTo";
  return RelationshipType2;
})(RelationshipType || {});
var SourceType = /* @__PURE__ */ ((SourceType2) => {
  SourceType2["AcademicPaper"] = "AcademicPaper";
  SourceType2["Book"] = "Book";
  SourceType2["NewsArticle"] = "NewsArticle";
  SourceType2["BlogPost"] = "BlogPost";
  SourceType2["WebPage"] = "WebPage";
  SourceType2["Video"] = "Video";
  SourceType2["Audio"] = "Audio";
  SourceType2["SocialMedia"] = "SocialMedia";
  SourceType2["Conversation"] = "Conversation";
  SourceType2["PersonalExperience"] = "PersonalExperience";
  SourceType2["OfficialDocument"] = "OfficialDocument";
  SourceType2["Dataset"] = "Dataset";
  SourceType2["LucidThought"] = "LucidThought";
  SourceType2["KnowledgeBase"] = "KnowledgeBase";
  SourceType2["Other"] = "Other";
  return SourceType2;
})(SourceType || {});
var CitationRelationship = /* @__PURE__ */ ((CitationRelationship2) => {
  CitationRelationship2["Supports"] = "Supports";
  CitationRelationship2["Refutes"] = "Refutes";
  CitationRelationship2["Context"] = "Context";
  CitationRelationship2["Origin"] = "Origin";
  CitationRelationship2["Alternative"] = "Alternative";
  CitationRelationship2["Methodology"] = "Methodology";
  CitationRelationship2["Definition"] = "Definition";
  CitationRelationship2["Reference"] = "Reference";
  return CitationRelationship2;
})(CitationRelationship || {});
var ContradictionType = /* @__PURE__ */ ((ContradictionType2) => {
  ContradictionType2["Logical"] = "Logical";
  ContradictionType2["Factual"] = "Factual";
  ContradictionType2["Temporal"] = "Temporal";
  ContradictionType2["SourceConflict"] = "SourceConflict";
  ContradictionType2["ConfidenceInconsistency"] = "ConfidenceInconsistency";
  return ContradictionType2;
})(ContradictionType || {});
var InferenceType = /* @__PURE__ */ ((InferenceType2) => {
  InferenceType2["Deduction"] = "Deduction";
  InferenceType2["Induction"] = "Induction";
  InferenceType2["Abduction"] = "Abduction";
  InferenceType2["Analogy"] = "Analogy";
  return InferenceType2;
})(InferenceType || {});
var FederationStatus = /* @__PURE__ */ ((FederationStatus2) => {
  FederationStatus2["Pending"] = "Pending";
  FederationStatus2["Active"] = "Active";
  FederationStatus2["Revoked"] = "Revoked";
  FederationStatus2["Failed"] = "Failed";
  return FederationStatus2;
})(FederationStatus || {});
var BeliefStance = /* @__PURE__ */ ((BeliefStance2) => {
  BeliefStance2["StronglyAgree"] = "StronglyAgree";
  BeliefStance2["Agree"] = "Agree";
  BeliefStance2["Neutral"] = "Neutral";
  BeliefStance2["Disagree"] = "Disagree";
  BeliefStance2["StronglyDisagree"] = "StronglyDisagree";
  return BeliefStance2;
})(BeliefStance || {});
var ValidationVoteType = /* @__PURE__ */ ((ValidationVoteType2) => {
  ValidationVoteType2["Corroborate"] = "Corroborate";
  ValidationVoteType2["Contradict"] = "Contradict";
  ValidationVoteType2["Plausible"] = "Plausible";
  ValidationVoteType2["Implausible"] = "Implausible";
  ValidationVoteType2["Abstain"] = "Abstain";
  return ValidationVoteType2;
})(ValidationVoteType || {});
var ConsensusType = /* @__PURE__ */ ((ConsensusType2) => {
  ConsensusType2["StrongConsensus"] = "StrongConsensus";
  ConsensusType2["ModerateConsensus"] = "ModerateConsensus";
  ConsensusType2["WeakConsensus"] = "WeakConsensus";
  ConsensusType2["Contested"] = "Contested";
  ConsensusType2["Insufficient"] = "Insufficient";
  return ConsensusType2;
})(ConsensusType || {});
var PatternType = /* @__PURE__ */ ((PatternType2) => {
  PatternType2["Convergence"] = "Convergence";
  PatternType2["Divergence"] = "Divergence";
  PatternType2["Trend"] = "Trend";
  PatternType2["Cluster"] = "Cluster";
  PatternType2["ContradictionCluster"] = "ContradictionCluster";
  return PatternType2;
})(PatternType || {});
var RelationshipStage = /* @__PURE__ */ ((RelationshipStage2) => {
  RelationshipStage2["NoRelation"] = "NoRelation";
  RelationshipStage2["Acquaintance"] = "Acquaintance";
  RelationshipStage2["Collaborator"] = "Collaborator";
  RelationshipStage2["TrustedPeer"] = "TrustedPeer";
  RelationshipStage2["PartnerInTruth"] = "PartnerInTruth";
  return RelationshipStage2;
})(RelationshipStage || {});

// src/index.ts
var LucidClient = class {
  constructor(client, roleName = "lucid") {
    this.client = client;
    this.roleName = roleName;
    this.lucid = new LucidZomeClient(client, roleName, "lucid");
    this.sources = new SourcesZomeClient(client, roleName, "sources");
    this.temporal = new TemporalZomeClient(client, roleName, "temporal");
    this.privacy = new PrivacyZomeClient(client, roleName, "privacy");
    this.reasoning = new ReasoningZomeClient(client, roleName, "reasoning");
    this.bridge = new BridgeZomeClient(client, roleName, "bridge");
    this.collective = new CollectiveZomeClient(client, roleName, "collective");
  }
  // ============================================================================
  // CONVENIENCE METHODS (delegate to lucid zome)
  // ============================================================================
  /** Create a new thought */
  async createThought(...args) {
    return this.lucid.createThought(...args);
  }
  /** Get a thought by ID */
  async getThought(...args) {
    return this.lucid.getThought(...args);
  }
  /** Update an existing thought */
  async updateThought(...args) {
    return this.lucid.updateThought(...args);
  }
  /** Delete a thought */
  async deleteThought(...args) {
    return this.lucid.deleteThought(...args);
  }
  /** Get all thoughts for the current user */
  async getMyThoughts() {
    return this.lucid.getMyThoughts();
  }
  /** Search thoughts with filters */
  async searchThoughts(...args) {
    return this.lucid.searchThoughts(...args);
  }
  /** Get statistics about the knowledge graph */
  async getStats() {
    return this.lucid.getStats();
  }
  // ============================================================================
  // HIGH-LEVEL OPERATIONS
  // ============================================================================
  /**
   * Create a thought with a source citation in one call.
   */
  async createThoughtWithSource(thoughtInput, sourceInput, citationOptions) {
    const thought = await this.lucid.createThought(thoughtInput);
    const source = await this.sources.createSource(sourceInput);
    const citation = await this.sources.createCitation({
      thought_id: thought.id,
      source_id: source.id,
      location: citationOptions?.location,
      quote: citationOptions?.quote,
      relationship: citationOptions?.relationship ?? "Supports" /* Supports */
    });
    return { thought, source, citation };
  }
  /**
   * Get a thought with its full context (sources, relationships, history).
   */
  async getThoughtWithContext(thoughtId) {
    const [thought, sources, relationships, history, contradictions] = await Promise.all([
      this.lucid.getThought(thoughtId),
      this.sources.getThoughtSources(thoughtId),
      this.lucid.getThoughtRelationships(thoughtId),
      this.temporal.getThoughtHistory(thoughtId),
      this.reasoning.getThoughtContradictions(thoughtId)
    ]);
    return {
      thought,
      sources,
      relationships,
      history,
      contradictions
    };
  }
  /**
   * Share a thought with specific agents.
   */
  async shareThought(thoughtId, agents, permissions = ["read"]) {
    const grants = await Promise.all(
      agents.map(
        (agent) => this.privacy.grantAccess({
          thought_id: thoughtId,
          grantee: agent,
          permissions
        })
      )
    );
    return grants;
  }
  /**
   * Make a thought public.
   */
  async makePublic(thoughtId) {
    return this.privacy.setSharingPolicy({
      thought_id: thoughtId,
      visibility: { type: "Public" }
    });
  }
  /**
   * Make a thought private.
   */
  async makePrivate(thoughtId) {
    return this.privacy.setSharingPolicy({
      thought_id: thoughtId,
      visibility: { type: "Private" }
    });
  }
};
var index_default = LucidClient;
export {
  BeliefStance,
  BridgeZomeClient,
  CitationRelationship,
  CollectiveZomeClient,
  ConsensusType,
  ContradictionType,
  EmpiricalLevel,
  FederationStatus,
  HarmonicLevel,
  InferenceType,
  LucidClient,
  LucidZomeClient,
  MaterialityLevel,
  NormativeLevel,
  PatternType,
  PrivacyZomeClient,
  ReasoningZomeClient,
  RelationshipStage,
  RelationshipType,
  SOURCE_QUALITY_WEIGHTS,
  SourceType,
  SourcesZomeClient,
  TemporalZomeClient,
  ThoughtType,
  ValidationVoteType,
  calculateEpistemicStrength,
  decodeRecord,
  decodeRecords,
  index_default as default,
  epistemicCode,
  formatTimestamp,
  getSourceQualityWeight,
  parseToTimestamp
};
