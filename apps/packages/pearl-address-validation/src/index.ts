import { bech32m } from 'bech32';

enum Network {
  mainnet = 'mainnet',
  testnet = 'testnet',
  regtest = 'regtest',
  simnet = 'simnet',
}

enum AddressType {
  p2tr = 'p2tr',
  p2mr = 'p2mr',
}

type AddressInfo = {
  bech32: boolean;
  network: Network;
  address: string;
  type: AddressType;
};

type Options = {
  castTestnetTo?: Network.regtest | Network.simnet;
};

const prefixToNetwork: { [key: string]: Network } = {
  prl: Network.mainnet,
  tprl: Network.testnet,
  rprl: Network.simnet,
};

const witnessVersionToType: { [key: number]: AddressType } = {
  1: AddressType.p2tr,
  2: AddressType.p2mr,
};

function castTestnetTo(fromNetwork: Network, toNetwork?: Network.regtest | Network.simnet): Network {
  if (!toNetwork) {
    return fromNetwork;
  }

  if (fromNetwork === Network.mainnet) {
    throw new Error('Cannot cast mainnet to non-mainnet');
  }

  return toNetwork;
}

const normalizeAddressInfo = (addressInfo: AddressInfo, options?: Options): AddressInfo => {
  return {
    ...addressInfo,
    network: castTestnetTo(addressInfo.network, options?.castTestnetTo),
  };
};

const getAddressInfo = (address: string, options?: Options): AddressInfo => {
  let decoded;

  try {
    decoded = bech32m.decode(address);
  } catch (error) {
    throw new Error('Invalid address');
  }

  const network = prefixToNetwork[decoded.prefix];

  if (network === undefined) {
    throw new Error('Invalid address');
  }

  const witnessVersion = decoded.words[0];

  if (witnessVersion === undefined) {
    throw new Error('Invalid address');
  }

  const type = witnessVersionToType[witnessVersion];

  if (type === undefined) {
    throw new Error('Invalid address');
  }

  const program = bech32m.fromWords(decoded.words.slice(1));

  if (program.length !== 32) {
    throw new Error('Invalid address');
  }

  return normalizeAddressInfo(
    {
      bech32: true,
      network,
      address,
      type,
    },
    options,
  );
};

const validate = (address: string, network?: Network, options?: Options) => {
  try {
    const addressInfo = getAddressInfo(address, options);

    if (network) {
      return network === addressInfo.network;
    }

    return true;
  } catch (error) {
    return false;
  }
};

export { getAddressInfo, Network, AddressType, validate };
export type { AddressInfo };
export default validate;
