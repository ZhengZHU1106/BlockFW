// contracts/FirewallRules.sol
pragma solidity ^0.8.0;

contract FirewallRules {
    uint[] public blockedPorts;

    function addPort(uint port) public {
        blockedPorts.push(port);
    }

    function getPort(uint index) public view returns (uint) {
        return blockedPorts[index];
    }

    function getLength() public view returns (uint) {
        return blockedPorts.length;
    }
}
